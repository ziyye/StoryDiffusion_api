from this import d

import numpy as np
import torch
import gc
import copy
import os
import random
import datetime
from PIL import ImageFont
from utils.gradio_utils import (
    character_to_dict,
    process_original_prompt,
    get_ref_character,
    cal_attn_mask_xl,
    cal_attn_indice_xl_effcient_memory,
    is_torch2_available,
)

if is_torch2_available():
    from utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils import AttnProcessor
from huggingface_hub import hf_hub_download
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch.nn.functional as F
from diffusers.utils.loading_utils import load_image
from utils.utils import get_comic
from utils.style_template import styles
from utils.load_models_utils import get_models_dict, load_models

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Japanese Anime"
global models_dict

models_dict = get_models_dict()

# Automatically select the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"@@device:{device}")


# check if the file exists locally at a specified path before downloading it.
# if the file doesn't exist, it uses `hf_hub_download` to download the file
# and optionally move it to a specific directory. If the file already
# exists, it simply uses the local path.

# local_dir = "data/"
# photomaker_local_path = f"{local_dir}photomaker-v1.bin"
# if not os.path.exists(photomaker_local_path):
#     photomaker_path = hf_hub_download(
#         repo_id="TencentARC/PhotoMaker",
#         filename="photomaker-v1.bin",
#         repo_type="model",
#         local_dir=local_dir,
#     )
# else:
#     photomaker_path = photomaker_local_path
photomaker_path = '/gpfs/public/vl/linhualing/.cache/huggingface/hub/models--TencentARC--PhotoMaker/snapshots/f68f8e6309bf213d28d68230abff0ccc92de9f30/photomaker-v1.bin'

MAX_SEED = np.iinfo(np.int32).max


def setup_seed(seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



#################################################
def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted(
        [os.path.join(folder_name, basename) for basename in image_basename_list]
    )
    return image_path_list


#################################################
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        id_length=4,
        device=device,
        dtype=torch.float16,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # un_cond_hidden_states, cond_hidden_states = hidden_states.chunk(2)
        # un_cond_hidden_states = self.__call2__(attn, un_cond_hidden_states,encoder_hidden_states,attention_mask,temb)
        # 生成一个0到1之间的随机数
        global total_count, attn_count, cur_step, indices1024, indices4096
        global sa32, sa64
        global write
        global height, width
        global character_dict, character_index_dict, invert_character_index_dict, cur_character, ref_indexs_dict, ref_totals, cur_character
        if attn_count == 0 and cur_step == 0:
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )
        if write:
            assert len(cur_character) == 1
            if hidden_states.shape[1] == (height // 32) * (width // 32):
                indices = indices1024
            else:
                indices = indices4096
            # print(f"white:{cur_step}")
            total_batch_size, nums_token, channel = hidden_states.shape
            img_nums = total_batch_size // 2
            hidden_states = hidden_states.reshape(-1, img_nums, nums_token, channel)
            # print(img_nums,len(indices),hidden_states.shape,self.total_length)
            if cur_character[0] not in self.id_bank:
                self.id_bank[cur_character[0]] = {}
            self.id_bank[cur_character[0]][cur_step] = [
                hidden_states[:, img_ind, indices[img_ind], :]
                .reshape(2, -1, channel)
                .clone()
                for img_ind in range(img_nums)
            ]
            hidden_states = hidden_states.reshape(-1, nums_token, channel)
            # self.id_bank[cur_step] = [hidden_states[:self.id_length].clone(), hidden_states[self.id_length:].clone()]
        else:
            # encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),self.id_bank[cur_step][1].to(self.device)))
            # TODO: ADD Multipersion Control
            encoder_arr = []
            for character in cur_character:
                encoder_arr = encoder_arr + [
                    tensor.to(self.device)
                    for tensor in self.id_bank[character][cur_step]
                ]
        # 判断随机数是否大于0.5
        if cur_step < 1:
            hidden_states = self.__call2__(
                attn, hidden_states, None, attention_mask, temb
            )
        else:  # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            # print(f"hidden state shape {hidden_states.shape[1]}")
            if random_number > rand_num:
                if hidden_states.shape[1] == (height // 32) * (width // 32):
                    indices = indices1024
                else:
                    indices = indices4096
                # print("before attention",hidden_states.shape,attention_mask.shape,encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
                if write:
                    total_batch_size, nums_token, channel = hidden_states.shape
                    img_nums = total_batch_size // 2
                    hidden_states = hidden_states.reshape(
                        -1, img_nums, nums_token, channel
                    )
                    encoder_arr = [
                        hidden_states[:, img_ind, indices[img_ind], :].reshape(
                            2, -1, channel
                        )
                        for img_ind in range(img_nums)
                    ]
                    for img_ind in range(img_nums):
                        # print(img_nums)
                        # assert img_nums != 1
                        img_ind_list = [i for i in range(img_nums)]
                        # print(img_ind_list,img_ind)
                        img_ind_list.remove(img_ind)
                        # print(img_ind,invert_character_index_dict[img_ind])
                        # print(character_index_dict[invert_character_index_dict[img_ind]])
                        # print(img_ind_list)
                        # print(img_ind,img_ind_list)
                        encoder_hidden_states_tmp = torch.cat(
                            [encoder_arr[img_ind] for img_ind in img_ind_list]
                            + [hidden_states[:, img_ind, :, :]],
                            dim=1,
                        )

                        hidden_states[:, img_ind, :, :] = self.__call2__(
                            attn,
                            hidden_states[:, img_ind, :, :],
                            encoder_hidden_states_tmp,
                            None,
                            temb,
                        )
                else:
                    _, nums_token, channel = hidden_states.shape
                    # img_nums = total_batch_size // 2
                    # encoder_hidden_states = encoder_hidden_states.reshape(-1,img_nums,nums_token,channel)
                    hidden_states = hidden_states.reshape(2, -1, nums_token, channel)
                    # print(len(indices))
                    # encoder_arr = [encoder_hidden_states[:,img_ind,indices[img_ind],:].reshape(2,-1,channel) for img_ind in range(img_nums)]
                    encoder_hidden_states_tmp = torch.cat(
                        encoder_arr + [hidden_states[:, 0, :, :]], dim=1
                    )
                    # print(len(encoder_arr),encoder_hidden_states_tmp.shape)
                    hidden_states[:, 0, :, :] = self.__call2__(
                        attn,
                        hidden_states[:, 0, :, :],
                        encoder_hidden_states_tmp,
                        None,
                        temb,
                    )
                hidden_states = hidden_states.reshape(-1, nums_token, channel)
            else:
                hidden_states = self.__call2__(
                    attn, hidden_states, None, attention_mask, temb
                )
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            indices1024, indices4096 = cal_attn_indice_xl_effcient_memory(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )

        return hidden_states

    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        # else:
        #     encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_attention_processor(unet, id_length, is_ipadapter=False):
    global attn_procs
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
            else:
                attn_procs[name] = AttnProcessor()
        else:
            if is_ipadapter:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1,
                    num_tokens=4,
                ).to(unet.device, dtype=torch.float16)
            else:
                attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))


#################################################
#################################################
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

css = """
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
"""


def save_single_character_weights(unet, character, description, filepath):
    """
    保存 attention_processor 类中的 id_bank GPU Tensor 列表到指定文件中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重要保存到的文件路径。
    """
    weights_to_save = {}
    weights_to_save["description"] = description
    weights_to_save["character"] = character
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 将每个 Tensor 转到 CPU 并转为列表，以确保它可以被序列化
            weights_to_save[attn_name] = {}
            for step_key in attn_processor.id_bank[character].keys():
                weights_to_save[attn_name][step_key] = [
                    tensor.cpu()
                    for tensor in attn_processor.id_bank[character][step_key]
                ]
    # 使用torch.save保存权重
    torch.save(weights_to_save, filepath)


def load_single_character_weights(unet, filepath):
    """
    从指定文件中加载权重到 attention_processor 类的 id_bank 中。
    参数:
    - model: 包含 attention_processor 类实例的模型。
    - filepath: 权重文件的路径。
    """
    # 使用torch.load来读取权重
    weights_to_load = torch.load(filepath, map_location=torch.device("cpu"))
    character = weights_to_load["character"]
    description = weights_to_load["description"]
    for attn_name, attn_processor in unet.attn_processors.items():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            # 转移权重到GPU（如果GPU可用的话）并赋值给id_bank
            attn_processor.id_bank[character] = {}
            for step_key in weights_to_load[attn_name].keys():
                attn_processor.id_bank[character][step_key] = [
                    tensor.to(unet.device)
                    for tensor in weights_to_load[attn_name][step_key]
                ]


def save_results(unet, img_list):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"results/{timestamp}"
    weight_folder_name = f"{folder_name}/weights"
    # 创建文件夹
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(weight_folder_name)

    for idx, img in enumerate(img_list):
        file_path = os.path.join(folder_name, f"image_{idx}.png")  # 图片文件名
        if idx == 0:
            print(f"save image to {file_path}")
        img.save(file_path)
    global character_dict

    # save the characters features offline
    for char in character_dict:  # {'[Kitty]': 'a girl, wearing white shirt, black skirt, black tie, yellow hair'}
        description = character_dict[char]
        save_single_character_weights(unet, char, description, os.path.join(weight_folder_name, f'{char}.pt'))

    return os.path.abspath(folder_name)


#################################################
global attn_count, total_count, id_length, total_length, cur_step, cur_model_type
global write
global sa32, sa64
global height, width
attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
global attn_procs, unet
attn_procs = {}
###
write = False
###
sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
###
global pipe
global sd_model_path
pipe = None

use_kolors = False
if not use_kolors:
    sd_model_path = models_dict["Unstable"]["path"]  # "SG161222/RealVisXL_V4.0"
    single_files = models_dict["Unstable"]["single_files"]
    ### LOAD Stable Diffusion Pipeline
    if single_files:
        pipe = StableDiffusionXLPipeline.from_single_file(
            sd_model_path, torch_dtype=torch.float16
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            sd_model_path, torch_dtype=torch.float16, use_safetensors=False
        )
elif use_kolors:
    from diffusers import KolorsPipeline
    pipe = KolorsPipeline.from_pretrained(
        "Kwai-Kolors/Kolors-diffusers", 
        torch_dtype=torch.float16, 
        variant="fp16"
    )


pipe = pipe.to(device)
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
pipe.enable_vae_slicing()
# if device != "mps":
#     pipe.enable_model_cpu_offload()
unet = pipe.unet
cur_model_type = "Unstable" + "-" + "original"
### Insert PairedAttention
for name in unet.attn_processors.keys():
    cross_attention_dim = (
        None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    )
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None and (name.startswith("up_blocks")):
        attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
        total_count += 1
    else:
        attn_procs[name] = AttnProcessor()
print("successsfully load paired self-attention")
print(f"number of the processor : {total_count}")
unet.set_attn_processor(copy.deepcopy(attn_procs))
global mask1024, mask4096
mask1024, mask4096 = cal_attn_mask_xl(
    total_length,
    id_length,
    sa32,
    sa64,
    height,
    width,
    device=device,
    dtype=torch.float16,
)



def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [
        p.replace("{prompt}", positive) for positive in positives
    ], n + " " + negative


def load_character_files(character_files: str):
    if character_files == "":
        raise Exception("Please set a character file!")
    character_files_arr = character_files.splitlines()
    primarytext = []
    for character_file_name in character_files_arr:
        character_file = torch.load(
            character_file_name, map_location=torch.device("cpu")
        )
        primarytext.append(character_file["character"] + character_file["description"])
    return array2string(primarytext)


def load_character_files_on_running(unet, character_files: str):
    if character_files == "":
        return False
    character_files_arr = character_files.splitlines()
    for character_file in character_files_arr:
        load_single_character_weights(unet, character_file)
    return True


######### Image Generation ##############
def process_generation(
    _sd_type,
    _model_type,
    _upload_images,
    _num_steps,
    style_name,
    _Ip_Adapter_Strength,
    _style_strength_ratio,
    guidance_scale,
    seed_,
    sa32_,
    sa64_,
    id_length_,
    general_prompt,
    negative_prompt,
    prompt_array,
    G_height,
    G_width,
    _comic_type,
    font_choice,
    _char_files,
):  # Corrected font_choice usage
    if len(general_prompt.splitlines()) >= 10:  # limit of characters number
        raise Exception(
            "Support for more than three characters is temporarily unavailable due to VRAM limitations, but this issue will be resolved soon."
        )
    _model_type = "Photomaker" if _model_type == "Using Ref Images" else "original"
    if _model_type == "Photomaker" and "img" not in general_prompt:
        raise Exception(
            'Please add the triger word " img "  behind the class word you want to customize, such as: man img or woman img'
        )
    if _upload_images is None and _model_type != "original":
        raise Exception(f"Cannot find any input face image!")
    global sa32, sa64, id_length, total_length, attn_procs, unet, cur_model_type
    global write
    global cur_step, attn_count
    global height, width
    height = G_height
    width = G_width
    global pipe
    global sd_model_path, models_dict
    sd_model_path = models_dict[_sd_type]
    use_safe_tensor = True
    for attn_processor in pipe.unet.attn_processors.values():
        if isinstance(attn_processor, SpatialAttnProcessor2_0):
            for values in attn_processor.id_bank.values():
                del values
            attn_processor.id_bank = {}
            attn_processor.id_length = id_length
            attn_processor.total_length = id_length + 1
    gc.collect()
    torch.cuda.empty_cache()
    if cur_model_type != _sd_type + "-" + _model_type:
        # apply the style template
        ##### load pipe
        del pipe
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        model_info = models_dict[_sd_type]
        model_info["model_type"] = _model_type
        pipe = load_models(model_info, device=device, photomaker_path=photomaker_path)
        set_attention_processor(pipe.unet, id_length_, is_ipadapter=False)
        ##### ########################
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        cur_model_type = _sd_type + "-" + _model_type
        pipe.enable_vae_slicing()
        if device != "mps":
            pipe.enable_model_cpu_offload()
    else:
        unet = pipe.unet
        # unet.set_attn_processor(copy.deepcopy(attn_procs))

    load_chars = load_character_files_on_running(unet, character_files=_char_files)

    prompts = prompt_array.splitlines()
    global character_dict, character_index_dict, invert_character_index_dict, ref_indexs_dict, ref_totals
    character_dict, character_list = character_to_dict(general_prompt)

    start_merge_step = int(float(_style_strength_ratio) / 100 * _num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"start_merge_step:{start_merge_step}")
    generator = torch.Generator(device=device).manual_seed(seed_)
    sa32, sa64 = sa32_, sa64_
    id_length = id_length_
    clipped_prompts = prompts[:]
    nc_indexs = []
    for ind, prompt in enumerate(clipped_prompts):
        if "[NC]" in prompt:
            nc_indexs.append(ind)
            if ind < id_length:
                raise Exception(
                    f"The first {id_length} row is id prompts, cannot use [NC]!"
                )
    prompts = [
        prompt if "[NC]" not in prompt else prompt.replace("[NC]", "")
        for prompt in clipped_prompts
    ]

    prompts = [
        prompt.rpartition("#")[0] if "#" in prompt else prompt for prompt in prompts
    ]
    print(prompts)
    # id_prompts = prompts[:id_length]
    (
        character_index_dict,
        invert_character_index_dict,
        replace_prompts,
        ref_indexs_dict,
        ref_totals,
    ) = process_original_prompt(character_dict, prompts.copy(), id_length)
    if _model_type != "original":
        input_id_images_dict = {}
        if len(_upload_images) != len(character_dict.keys()):
            raise Exception(
                f"You upload images({len(_upload_images)}) is not equal to the number of characters({len(character_dict.keys())})!"
            )
        for ind, img in enumerate(_upload_images):
            input_id_images_dict[character_list[ind]] = [load_image(img)]
    print(character_dict)
    print(character_index_dict)
    print(invert_character_index_dict)
    # real_prompts = prompts[id_length:]
    if device == "cuda":
        torch.cuda.empty_cache()
    write = True
    cur_step = 0

    attn_count = 0
    # id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
    # print(id_prompts)
    setup_seed(seed_)
    total_results = []
    id_images = []
    results_dict = {}
    global cur_character
    if not load_chars:
        print("do not load characters")
        for character_key in character_dict.keys():
            cur_character = [character_key]
            ref_indexs = ref_indexs_dict[character_key]
            print(character_key, ref_indexs)
            current_prompts = [replace_prompts[ref_ind] for ref_ind in ref_indexs]
            print(current_prompts)
            setup_seed(seed_)
            generator = torch.Generator(device=device).manual_seed(seed_)
            cur_step = 0
            cur_positive_prompts, negative_prompt = apply_style(
                style_name, current_prompts, negative_prompt
            )
            if use_kolors:
                if type(cur_positive_prompts) == list and type(negative_prompt) == str:
                    negative_prompt = [negative_prompt] * len(cur_positive_prompts)
                elif type(cur_positive_prompts) == str and type(negative_prompt) == list:
                    negative_prompt = negative_prompt[0]
            print(f"cur_positive_prompts is {cur_positive_prompts}")
            if _model_type == "original":
                id_images = pipe(
                    cur_positive_prompts,
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=generator,
                ).images
            elif _model_type == "Photomaker":
                id_images = pipe(
                    cur_positive_prompts,
                    input_id_images=input_id_images_dict[character_key],
                    num_inference_steps=_num_steps,
                    guidance_scale=guidance_scale,
                    start_merge_step=start_merge_step,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    generator=generator,
                ).images
            else:
                raise NotImplementedError(
                    "You should choice between original and Photomaker!",
                    f"But you choice {_model_type}",
                )

            # total_results = id_images + total_results
            # yield total_results
            print(id_images)
            for ind, img in enumerate(id_images):
                print(ref_indexs[ind])
                results_dict[ref_indexs[ind]] = img
            # real_images = []
            yield [results_dict[ind] for ind in results_dict.keys()]
    write = False
    if not load_chars:
        real_prompts_inds = [
            ind for ind in range(len(prompts)) if ind not in ref_totals
        ]
    else:
        real_prompts_inds = [ind for ind in range(len(prompts))]
    print(real_prompts_inds)

    for real_prompts_ind in real_prompts_inds:
        real_prompt = replace_prompts[real_prompts_ind]
        cur_character = get_ref_character(prompts[real_prompts_ind], character_dict)
        print(cur_character, real_prompt)
        setup_seed(seed_)
        if len(cur_character) > 1 and _model_type == "Photomaker":
            raise Exception(
                "Temporarily Not Support Multiple character in Ref Image Mode!"
            )
        generator = torch.Generator(device=device).manual_seed(seed_)
        cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        if use_kolors:
            if type(real_prompt) == list and type(negative_prompt) == str:
                negative_prompt = [negative_prompt] * len(real_prompt)
            elif type(real_prompt) == str and type(negative_prompt) == list:
                negative_prompt = negative_prompt[0]
        print(f"real_prompt is {real_prompt}")
        if _model_type == "original":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]
        elif _model_type == "Photomaker":
            results_dict[real_prompts_ind] = pipe(
                real_prompt,
                input_id_images=(
                    input_id_images_dict[cur_character[0]]
                    if real_prompts_ind not in nc_indexs
                    else input_id_images_dict[character_list[0]]
                ),
                num_inference_steps=_num_steps,
                guidance_scale=guidance_scale,
                start_merge_step=start_merge_step,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
                nc_flag=True if real_prompts_ind in nc_indexs else False,
            ).images[0]
        else:
            raise NotImplementedError(
                "You should choice between original and Photomaker!",
                f"But you choice {_model_type}",
            )
        yield [results_dict[ind] for ind in results_dict.keys()]
    total_results = [results_dict[ind] for ind in range(len(prompts))]
    if _comic_type != "No typesetting (default)":
        captions = prompt_array.splitlines()
        captions = [caption.replace("[NC]", "") for caption in captions]
        captions = [
            caption.split("#")[-1] if "#" in caption else caption
            for caption in captions
        ]
        font_path = os.path.join("fonts", font_choice)
        font = ImageFont.truetype(font_path, int(45))
        total_results = (
            get_comic(total_results, _comic_type, captions=captions, font=font)
            + total_results
        )  # a list of 'PIL.Image.Image'
    save_root = save_results(pipe.unet, total_results)

    # yield total_results
    yield save_root


def array2string(arr):
    stringtmp = ""
    for i, part in enumerate(arr):
        if i != len(arr) - 1:
            stringtmp += part + "\n"
        else:
            stringtmp += part

    return stringtmp


#################################################

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import threading
import socket
import os
import datetime
import sys
import time


import requests
from io import BytesIO

from pydantic import BaseModel, ConfigDict

class StoryDiffusionInputData(BaseModel):
    model_config = ConfigDict(underscore_attrs_are_private=False)

    sd_type: str = 'Unstable'
    model_type: str = 'original'
    upload_images: list[str]
    num_steps: int = 20
    style_name: str = 'Japanese Anime'
    Ip_Adapter_Strength: float = 0.5
    style_strength_ratio: float = 20
    guidance_scale: float = 5
    seed_: int = 1
    sa32_: float = 0.3
    sa64_: float = 0.5
    id_length_: int = 3
    general_prompt: str = '[Kitty]a girl',
    negative_prompt: str = 'bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs'
    prompt_array: str = '[Kitty]running on a beach\n[Kitty]playing computer games\n[Kitty] eat rice'
    G_height: int = 768
    G_width: int = 768
    comic_type: str = 'Classic Comic Style'
    font_choice: str = 'Inkfree.ttf'
    char_files: str
    
    response_format: str = "base64"  # "b64_json", "tosUrl", "pickle"


def do_infer_1(generate_cfg):
    result_pil_imgs = []
    for pil_img in process_generation(**generate_cfg):
        if type(pil_img) == list:
            print(f"return {len(pil_img)} images, sizes {[img.size for img in pil_img]}")
            result_pil_imgs += pil_img
        else:
            print(f"return image size {pil_img.size}")
            result_pil_imgs.append(pil_img)
    return result_pil_imgs


def do_infer(generate_cfg):
    for x in process_generation(**generate_cfg):
        if type(x) is str and os.path.isdir(x):
            # the last value yield is save_root path of the images and weights
            return x


app = FastAPI()

print("API is starting up")
lock = threading.Lock()


@app.post("/story_storydiffusion")
def story_storydiffusion(data: StoryDiffusionInputData):
    try:
        print("Entered story_storydiffusion function")
        print(f"Received request with data: {str(data)}")

        with lock:
            print("Acquired lock, starting inference")

            generate_cfg = data.model_dump(include=data.__dict__.keys())
            
            generate_cfg = {
                "_sd_type": data.sd_type,
                "_model_type": data.model_type,
                "_upload_images": data.upload_images,
                "_num_steps": data.num_steps,
                "style_name": data.style_name,
                "_Ip_Adapter_Strength": data.Ip_Adapter_Strength,
                "_style_strength_ratio": data.style_strength_ratio,
                "guidance_scale": data.guidance_scale,
                "seed_": data.seed_,
                "sa32_": data.sa32_,
                "sa64_": data.sa64_,
                "id_length_": data.id_length_,
                "general_prompt": data.general_prompt,
                "negative_prompt": data.negative_prompt,
                "prompt_array": data.prompt_array,
                "G_height": data.G_height,
                "G_width": data.G_width,
                "_comic_type": data.comic_type,
                "font_choice": data.font_choice,
                "_char_files": data.char_files
            }
            
            result = do_infer(generate_cfg)
            print("Inference complete")

        if data.response_format == "save_root":
            print(f"Save root of the result images and character weights is {result}")
            return JSONResponse(content={"success": True, "data": result})
        else:
            return JSONResponse(content={"success": False, "data": "Unsupported response format"})
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return JSONResponse(content={"success": False, "data": str(e)})

print("API is ready to accept requests")


import uvicorn

if __name__ == "__main__":
    # config = uvicorn.Config("api_flux:app", host="0.0.0.0", port=8977, access_log=True, workers=1)
    # server = uvicorn.Server(config)
    # server.run()
    uvicorn.run(app, host="0.0.0.0", port=8977)


# source /gpfs/public/vl/wangjiazhi/.bashrc; conda activate storydiffusion
# cd /gpfs/public/vl/wangjiazhi/00_projs/StoryDiffusion
# python gradio_app_sdxl_specific_id_low_vram_api.py

# uvicorn gradio_app_sdxl_specific_id_low_vram_api:app --host 0.0.0.0 --port 8977
