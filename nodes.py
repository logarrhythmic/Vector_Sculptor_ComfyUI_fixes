import torch
import comfy.model_management as model_management
from copy import deepcopy

def maximum_absolute_values(tensors,reversed=False):
    shape = tensors.shape
    tensors = tensors.reshape(shape[0], -1)
    tensors_abs = torch.abs(tensors)
    if not reversed:
        max_abs_idx = torch.argmax(tensors_abs, dim=0)
    else:
        max_abs_idx = torch.argmin(tensors_abs, dim=0)
    result = tensors[max_abs_idx, torch.arange(tensors.shape[1])]
    return result.reshape(shape[1:])

def get_closest_token_cosine_similarities(single_embedding, all_embeddings, return_scores=False):
    cosS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cosS(all_embeddings, single_embedding.unsqueeze(0).to(all_embeddings.device))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_id_list = sorted_ids.tolist()
    if not return_scores:
        return best_id_list
    scores_list = sorted_scores.tolist()
    return best_id_list, scores_list

def get_single_cosine_score(single_embedding, other_embedding):
    cosS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    score = cosS(other_embedding.unsqueeze(0), single_embedding.unsqueeze(0).to(other_embedding.device)).item()
    return score

def refine_token_weight(token_id, all_embeddings, sculptor_method, sculptor_multiplier):
    return refine_embedding_weight(all_embeddings[token_id], all_embeddings, sculptor_method, sculptor_multiplier, original_token_id=token_id)

# returns the sculpted embedding in cpu memory, initial_weight and all_weights should be on the same device but not necessarily 
def refine_embedding_weight(initial_embedding, all_embeddings, sculptor_method, sculptor_multiplier, original_token_id=None):
    init_mag = torch.linalg.vector_norm(initial_embedding)
    score_token_ids, scores = get_closest_token_cosine_similarities(initial_embedding,all_embeddings,True)
    
    if original_token_id and score_token_ids[0] == original_token_id:
        score_token_ids, scores = score_token_ids[1:], scores[1:]
    
    previous_cos_score = 0
    i = 0
    cos_score = 1
    near_scores = []
    near_region = []
    #ini_w = torch.clone(initial_weight)
    
    while cos_score > previous_cos_score:
        print(f"{i}: {previous_cos_score} -> {cos_score}")
        if i > 0:
            previous_cos_score = cos_score
        # get the actual embedding the current score is for
        candidate_embedding = all_embeddings[score_token_ids[i]]
        # add it to the embeddings included in the nearby region
        near_region.append(candidate_embedding)
        # add the current element of the scores array to the near region score list
        near_scores.append(scores[i])
        # get the element-wise sum of the near region embeddings
        near_region_sum = torch.sum(torch.stack(near_region),dim=0)
        # update the cosine similarity between the initial embedding and the nearby region sum
        cos_score = get_single_cosine_score(initial_embedding, near_region_sum)
        i += 1

    print(f"cos_score stopped increasing at {i-1}: {previous_cos_score} -> {cos_score}")
    
    # take the last element out of these
    del near_scores[-1]
    del near_region[-1]

    if len(near_region) <= 1: return initial_embedding.cpu(), 0

    if sculptor_method == "maximum_absolute":
        near_region_normalized = torch.stack([initial_embedding.div(init_mag)]+[t.div(torch.linalg.vector_norm(t)) for i, t in enumerate(near_region)])
        sculpted_embedding = maximum_absolute_values(near_region_normalized)
    elif sculptor_method == "add_minimum_absolute":
        near_region_normalized = torch.stack([initial_embedding.div(init_mag)]+[t.div(torch.linalg.vector_norm(t)) for i, t in enumerate(near_region)])
        near_region_min = maximum_absolute_values(near_region_normalized, True)
        sculpted_embedding = initial_embedding + near_region_min.mul(sculptor_multiplier)
    else:
        near_region_sum_scoresquared = torch.sum(torch.stack([t.mul(near_scores[i]**2) for i, t in enumerate(near_region)]), dim=0)
        final_score = get_single_cosine_score(initial_embedding, near_region_sum_scoresquared)
        print(f"final score: {final_score}")

        sculpting = near_region_sum_scoresquared.mul(final_score).mul(sculptor_multiplier)
        
        if sculptor_method == "backward":
            sculpted_embedding = initial_embedding + sculpting
        elif sculptor_method == "forward":
            sculpted_embedding = initial_embedding - sculpting

    sculpted_embedding.mul_(init_mag / torch.linalg.vector_norm(sculpted_embedding)) 
    return sculpted_embedding.cpu(), len(near_scores)

def vector_sculptor_text(clip, text, sculptor_method, token_normalization, sculptor_multiplier):
    return vector_sculptor_tokens(clip, clip.tokenize(text), sculptor_method, token_normalization, sculptor_multiplier)

def vector_sculptor_tokens(clip, tokens, sculptor_method, token_normalization, sculptor_multiplier):
    ignored_token_ids = [49406, 49407, 0]
    total_found = 0
    total_replaced = 0
    total_candidates = 0

    for stage_id in tokens:
        stage_found = 0
        stage_replaced = 0
        stage_candidates = 0
        mag_sum = 0
        mag_count = 0
        mag_coords = []
        if stage_id.lower() == "g":
            actual_multiplier = sculptor_multiplier * 4 / 1.5 #2048 to 768, this gives the same effect intensity on both CLIP
        else:
            actual_multiplier = sculptor_multiplier
        clip_model = getattr(clip.cond_stage_model, f"clip_{stage_id}", None)
        
        all_embeddings = torch.clone(clip_model.transformer.text_model.embeddings.token_embedding.weight).to(device=model_management.get_torch_device())
        if token_normalization == "mean of all tokens":
            all_mags = torch.stack([torch.linalg.vector_norm(t) for t in all_embeddings])
            mean_mag_all_embeddings = torch.mean(all_mags, dim=0).item()

        for x in range(len(tokens[stage_id])):
            for y in range(len(tokens[stage_id][x])):
                token, attn_weight = tokens[stage_id][x][y]

                if isinstance(token, torch.Tensor):
                    token_id = None
                    embedding = token
                else:
                    token_id = token
                    embedding = all_embeddings[token_id]

                if token_id not in ignored_token_ids and sculptor_multiplier > 0:
                    stage_candidates += 1
                    sculpted_embedding, n_found = refine_embedding_weight(embedding, all_embeddings, sculptor_method, actual_multiplier, original_token_id=token_id)
                    if n_found > 0:
                        stage_found += n_found
                        stage_replaced += 1
                else:
                    sculpted_embedding = embedding

                if token_normalization != "none" and y != 0 and token_id != 2:
                    mag = torch.linalg.vector_norm(sculpted_embedding)
                    match token_normalization:
                        case "mean" | "mean * attention":
                            mag_sum += mag.item()
                            mag_count += 1
                            mag_coords.append([x,y])
                        case "default * attention":
                            sculpted_embedding.mul_(attn_weight)
                        case "set at 1":
                            sculpted_embedding.div_(mag)
                        case "set at attention":
                            sculpted_embedding.mul_(attn_weight / mag)
                        case "mean of all tokens":
                            sculpted_embedding.mul_(mean_mag_all_embeddings / mag)
                tokens[stage_id][x][y] = (sculpted_embedding, attn_weight)

        if token_normalization is "mean" or "mean * attention" and mag_count > 0:
            mean_mag = mag_sum / mag_count
            for x, y in mag_coords:
                emb, attn_weight = tokens[stage_id][x][y]
                emb.mul_(mean_mag / torch.linalg.vector_norm(emb))
                if token_normalization == "mean * attention":
                    emb.mul_(attn_weight)
                tokens[stage_id][x][y] = (emb, attn_weight)

        del all_embeddings

        if stage_candidates > 0:
            print(f"stage_id: {stage_id} || total_found: {stage_found} / total_replaced: {stage_replaced} / total_candidates: {stage_candidates} / candidate proportion replaced: {round(100*stage_replaced/stage_candidates,2)}%")
    return tokens

class vector_sculptor_node:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text": ("STRING", {"multiline": True}),
                "sculptor_intensity": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.01}),
                "sculptor_method" : (["forward","backward","maximum_absolute","add_minimum_absolute"],),
                "token_normalization": (["none", "mean", "set at 1", "default * attention", "mean * attention", "set at attention", "mean of all tokens"],),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING","STRING",)
    RETURN_NAMES = ("Conditioning","Parameters_as_string",)
    CATEGORY = "conditioning"

    def exec(self, clip, text, sculptor_intensity, sculptor_method, token_normalization):
        sculptor_tokens = vector_sculptor_text(clip, text, sculptor_method, token_normalization, sculptor_intensity)
        cond, pooled = clip.encode_from_tokens(sculptor_tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]
        if sculptor_intensity == 0 and token_normalization == "none":
            parameters_as_string = "Disabled"
        else:
            parameters_as_string = f"Intensity: {round(sculptor_intensity,2)}\nMethod: {sculptor_method}\nNormalization: {token_normalization}"
        return (conditioning,parameters_as_string,)

def add_to_first_if_shorter(conditioning1,conditioning2,x=0):
    min_dim = min(conditioning1[x][0].shape[1],conditioning2[x][0].shape[1])
    if conditioning2[x][0].shape[1]>conditioning1[x][0].shape[1]:
        conditioning2[x][0][:,:min_dim,...] = conditioning1[x][0][:,:min_dim,...]
        conditioning1 = conditioning2
    return conditioning1

# cheap slerp / I will bet an eternity doing regex that this is the dark souls 2 camera direction formula
def average_and_keep_mag(v1,v2,p1):
    m1 = torch.linalg.vector_norm(v1)
    m2 = torch.linalg.vector_norm(v2)
    v0 = v1 * p1 + v2 * (1 - p1)
    v0 = v0 / torch.linalg.vector_norm(v0) * (m1 * p1 + m2 * (1 - p1))
    return v0

# from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
def slerp(high, low, val):
    dims = low.shape

    #flatten to batches
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.linalg.vector_norm(low, dim=1, keepdim=True)
    high_norm = high/torch.linalg.vector_norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)
    
class slerp_cond_node:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_to": ("CONDITIONING",),
                "conditioning_from": ("CONDITIONING",),
                "conditioning_to_strength": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, conditioning_to, conditioning_from,conditioning_to_strength):
        cond1 = deepcopy(conditioning_to)
        cond2 = deepcopy(conditioning_from)
        for x in range(min(len(cond1),len(cond2))):
            min_dim = min(cond1[x][0].shape[1],cond2[x][0].shape[1])
            if cond1[x][0].shape[2] == 2048:
                cond1[x][0][:,:min_dim,:768] = slerp(cond1[x][0][:,:min_dim,:768], cond2[x][0][:,:min_dim,:768], conditioning_to_strength)
                cond1[x][0][:,:min_dim,768:] = slerp(cond1[x][0][:,:min_dim,768:], cond2[x][0][:,:min_dim,768:], conditioning_to_strength)
            else:
                cond1[x][0][:,:min_dim,...] = slerp(cond1[x][0][:,:min_dim,...], cond2[x][0][:,:min_dim,...], conditioning_to_strength)
            cond1 = add_to_first_if_shorter(cond1,cond2,x)
        return (cond1,)

class average_keep_mag_node:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_to": ("CONDITIONING",),
                "conditioning_from": ("CONDITIONING",),
                "conditioning_to_strength": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, conditioning_to, conditioning_from,conditioning_to_strength):
        cond1 = deepcopy(conditioning_to)
        cond2 = deepcopy(conditioning_from)
        for x in range(min(len(cond1),len(cond2))):
            min_dim = min(cond1[x][0].shape[1],cond2[x][0].shape[1])
            if cond1[x][0].shape[2] == 2048:
                cond1[x][0][:,:min_dim,:768] = average_and_keep_mag(cond1[x][0][:,:min_dim,:768], cond2[x][0][:,:min_dim,:768], conditioning_to_strength)
                cond1[x][0][:,:min_dim,768:] = average_and_keep_mag(cond1[x][0][:,:min_dim,768:], cond2[x][0][:,:min_dim,768:], conditioning_to_strength)
            else:
                cond1[x][0][:,:min_dim,...] = average_and_keep_mag(cond1[x][0][:,:min_dim,...], cond2[x][0][:,:min_dim,...], conditioning_to_strength)
            cond1 = add_to_first_if_shorter(cond1,cond2,x)
        return (cond1,)
    
class norm_mag_node:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "empty_conditioning": ("CONDITIONING",),
                "enabled" : ("BOOLEAN", {"default": True}),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, conditioning, empty_conditioning, enabled):
        if not enabled: return (conditioning,)
        cond1 = deepcopy(conditioning)
        empty_cond = empty_conditioning[0][0]
        empty_tokens_no = empty_cond[0].shape[0]

        for x in range(len(cond1)):
            for y in range(len(cond1[x][0])):
                for z in range(len(cond1[x][0][y])):
                    if cond1[x][0][y][z].shape[0] == 2048:
                        cond1[x][0][y][z][:768] = cond1[x][0][y][z][:768]/torch.linalg.vector_norm(cond1[x][0][y][z][:768]) * torch.linalg.vector_norm(empty_cond[0][z%empty_tokens_no][:768])
                        cond1[x][0][y][z][768:] = cond1[x][0][y][z][768:]/torch.linalg.vector_norm(cond1[x][0][y][z][768:]) * torch.linalg.vector_norm(empty_cond[0][z%empty_tokens_no][768:])
                    else:
                        cond1[x][0][y][z] = cond1[x][0][y][z]/torch.linalg.vector_norm(cond1[x][0][y][z]) * torch.linalg.vector_norm(empty_cond[0][z%empty_tokens_no])
        return (cond1,)

class conditioning_merge_clip_g_l:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond_clip_l": ("CONDITIONING",),
                "cond_clip_g": ("CONDITIONING",),
            }
        }

    FUNCTION = "exec"
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "conditioning"

    def exec(self, cond_clip_l, cond_clip_g):
        conditioning_l = deepcopy(cond_clip_l)
        conditioning_g = deepcopy(cond_clip_g)
        for x in range(min(len(conditioning_g),len(conditioning_l))):
            min_dim = min(conditioning_g[x][0].shape[1],conditioning_l[x][0].shape[1])
            conditioning_g[x][0][:,:min_dim,:768] = conditioning_l[x][0][:,:min_dim,:768]
        return (conditioning_g,)
    
NODE_CLASS_MAPPINGS = {
    "CLIP Vector Sculptor text encode": vector_sculptor_node,
    "Conditioning (Slerp)": slerp_cond_node,
    "Conditioning (Average keep magnitude)": average_keep_mag_node,
    "Conditioning normalize magnitude to empty": norm_mag_node,
    "Conditioning SDXL merge clip_g / clip_l": conditioning_merge_clip_g_l,
}
