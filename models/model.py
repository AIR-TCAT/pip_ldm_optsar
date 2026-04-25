import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, AutoTokenizer, AutoModel

class SARPhysicsLatentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)

    def get_gradient(self, x):
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        grad_x = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return gradient.view(b, c, h, w)

    def gram_matrix(self, x):
        b, c, h, w = x.shape
        features = x.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w)

    def forward(self, pred_latents, target_latents):
        metrics = {}
        loss_sparse = torch.mean(torch.abs(pred_latents))
        pred_gram = self.gram_matrix(pred_latents)
        target_gram = self.gram_matrix(target_latents)
        loss_topo = F.mse_loss(pred_gram, target_gram)

        pred_grad = self.get_gradient(pred_latents)
        target_grad = self.get_gradient(target_latents)
        loss_geo = F.mse_loss(pred_grad, target_grad)

        abs_target = torch.abs(target_latents)
        mean_abs = torch.mean(abs_target, dim=[2, 3], keepdim=True)
        std_abs = torch.std(abs_target, dim=[2, 3], keepdim=True)
        k = 0.5
        tau = mean_abs + k * std_abs

        scattering_mask = (abs_target > tau).float()
        loss_scat = torch.mean(scattering_mask * (pred_latents - target_latents) ** 2)

        loss_stat = (
                F.mse_loss(pred_latents.mean(dim=[2, 3]), target_latents.mean(dim=[2, 3])) +
                F.mse_loss(pred_latents.std(dim=[2, 3]), target_latents.std(dim=[2, 3]))
        )

        metrics['loss_sparse'] = loss_sparse
        metrics['loss_topo'] = loss_topo
        metrics['loss_geo'] = loss_geo
        metrics['loss_scat'] = loss_scat
        metrics['loss_stat'] = loss_stat

        return metrics

class SemanticEnhancedTextEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Initializing SemanticEnhancedTextEncoder on {self.device}...")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("Your path of the model weight of clip-vit-base-patch32.")
        self.clip_text_encoder = CLIPTextModelWithProjection.from_pretrained("Your path of the model weight of clip-vit-base-patch32.").to(device)
        self.clip_text_encoder.eval()

        self.bert_tokenizer = AutoTokenizer.from_pretrained("Your path of the model weight of bert-base-uncased.")
        self.bert_model = AutoModel.from_pretrained("Your path of the model weight of bert-base-uncased.").to(device)
        self.bert_model.eval()

    def encode_text_with_semantic_hierarchy(self, text_description):
        with torch.no_grad():
            clip_inputs = self.clip_tokenizer(
                text_description, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(self.device)
            clip_embeddings = self.clip_text_encoder(**clip_inputs).text_embeds

            bert_inputs = self.bert_tokenizer(
                text_description, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(self.device)
            bert_outputs = self.bert_model(**bert_inputs)
            bert_embeddings = bert_outputs.last_hidden_state.mean(dim=1)

            combined_embeddings = torch.cat([clip_embeddings, bert_embeddings], dim=-1)
            return combined_embeddings

class SemanticGuidedControlNetModel(nn.Module):
    def __init__(self, controlnet, semantic_dim=1280):
        super().__init__()
        self.controlnet = controlnet
        self.semantic_projection = nn.Sequential(
            nn.Linear(semantic_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 768)
        )
        self.channel_configs = [320] * 4 + [640] * 3 + [1280] * 5
        self.semantic_condition_layers = nn.ModuleList([
            nn.Linear(768, channels) for channels in self.channel_configs
        ])
        self.mid_block_condition_layer = nn.Linear(768, 1280)

    def forward(self, x, timesteps, encoder_hidden_states, controlnet_cond, semantic_embeddings):
        semantic_features = self.semantic_projection(semantic_embeddings)
        batch_size = semantic_features.shape[0]

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x, timesteps, encoder_hidden_states, controlnet_cond, return_dict=False
        )

        enhanced_down_samples = []
        for i, sample in enumerate(down_block_res_samples):
            if i < len(self.semantic_condition_layers):
                if semantic_features.shape[0] != sample.shape[0]:
                    semantic_features_expanded = semantic_features.repeat(sample.shape[0] // batch_size, 1)
                else:
                    semantic_features_expanded = semantic_features

                semantic_cond = self.semantic_condition_layers[i](semantic_features_expanded)
                target_shape = (sample.shape[0], sample.shape[1], 1, 1)
                semantic_cond = semantic_cond.view(target_shape).expand_as(sample)
                enhanced_sample = sample + 0.1 * semantic_cond
                enhanced_down_samples.append(enhanced_sample)
            else:
                enhanced_down_samples.append(sample)

        if mid_block_res_sample is not None:
            if semantic_features.shape[0] != mid_block_res_sample.shape[0]:
                semantic_features_mid = semantic_features.repeat(mid_block_res_sample.shape[0] // batch_size, 1)
            else:
                semantic_features_mid = semantic_features

            semantic_cond_mid = self.mid_block_condition_layer(semantic_features_mid)
            semantic_cond_mid = semantic_cond_mid.view(
                mid_block_res_sample.shape[0], mid_block_res_sample.shape[1], 1, 1
            ).expand_as(mid_block_res_sample)
            enhanced_mid_sample = mid_block_res_sample + 0.1 * semantic_cond_mid
        else:
            enhanced_mid_sample = mid_block_res_sample

        return enhanced_down_samples, enhanced_mid_sample

def setup_semantic_lora_for_unet(unet, semantic_dim=1280, lora_rank=32, lora_alpha=64):
    class SemanticLoraAttnProcessor(nn.Module):
        def __init__(self, hidden_size, cross_attention_dim=None, rank=16):
            super().__init__()
            self.hidden_size = hidden_size
            self.cross_attention_dim = cross_attention_dim
            self.rank = rank
            self.scale = lora_alpha / rank
            self.to_q_lora_down = nn.Linear(hidden_size, rank, bias=False)
            self.to_q_lora_up = nn.Linear(rank, hidden_size, bias=False)
            ctx_dim = cross_attention_dim or hidden_size
            self.to_k_lora_down = nn.Linear(ctx_dim, rank, bias=False)
            self.to_k_lora_up = nn.Linear(rank, hidden_size, bias=False)
            self.to_v_lora_down = nn.Linear(ctx_dim, rank, bias=False)
            self.to_v_lora_up = nn.Linear(rank, hidden_size, bias=False)
            self.to_out_lora_down = nn.Linear(hidden_size, rank, bias=False)
            self.to_out_lora_up = nn.Linear(rank, hidden_size, bias=False)

        def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            query = attn.to_q(hidden_states)
            q_lora = self.to_q_lora_up(self.to_q_lora_down(hidden_states))
            query = query + self.scale * q_lora
            if encoder_hidden_states is None: encoder_hidden_states = hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            k_lora = self.to_k_lora_up(self.to_k_lora_down(encoder_hidden_states))
            v_lora = self.to_v_lora_up(self.to_v_lora_down(encoder_hidden_states))
            key = key + self.scale * k_lora
            value = value + self.scale * v_lora
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.to_out[0](hidden_states)
            out_lora = self.to_out_lora_up(self.to_out_lora_down(hidden_states))
            hidden_states = hidden_states + self.scale * out_lora
            return hidden_states

    for name, module in unet.named_modules():
        if "attn" in name and "processor" not in name:
            if hasattr(module, 'set_processor'):
                cross_attention_dim = module.to_k.in_features if hasattr(module, 'to_k') else None
                processor = SemanticLoraAttnProcessor(
                    module.to_q.in_features, cross_attention_dim, lora_rank
                )
                module.set_processor(processor)
    return unet