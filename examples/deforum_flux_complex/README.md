# Deforum Flux Complex Workflow

![image.png](attachment:42b6654e-741b-48ae-9bbd-3eabc3ddc952:image.png)

---

## Credits:

[Huemin:](https://huemin.art/) Creator of [Deforum](https://deforum.art/) and [Dream Computing](https://www.dreamcomputing.io/)

[Akatz:](https://akatz.ai/) Workflow builder and founder of [ComfyDock](https://comfydock.com/)

---

## Step 1. Download models

Required:

[flux1-dev-fp8.safetensors](https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors?download=true)

Optional (but recommended):

[Hyper-FLUX.1-dev-8steps-lora.safetensors](https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors?download=true)

[FLUX-dev-lora-AntiBlur.safetensors](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur/resolve/main/FLUX-dev-lora-AntiBlur.safetensors?download=true)

[Flux.1-Turbo-Alpha](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/tree/main)

---

## Step 2. Configure Inputs

![image.png](attachment:08179661-99fe-44a6-a5f8-01330618a7dc:image.png)

**Seed**: The initial seed used in generating

**Num Frames**: How many total frames the final animation should have

**Width**: The width of the images

**Height**: The height of the images

**Default Steps:** The default number of sampling steps that will be used if a step schedule is not present and enabled.

**Default Denoise:** The default denoise setting that will be used if a denoise schedule is not present and enabled.

**Default CFG:** The default Classifier Free Guidance setting that will be used if a CFG schedule is not present and enabled.

**Lora Loader Stack:** Where you can load FLUX loras to influence the final result.

**Use Upload Image?:** A switch that if set to TRUE will use the uploaded image as the initial frame in the generation sequence, and if FALSE will generate a random starting frame based on the prompt and seed.

---

## Step 3. Configure Prompt Schedules

![image.png](attachment:a1ea8b7d-418b-4c63-9459-b63be7e7d228:image.png)

The "Schedules" group has a Prompt Schedule node which will let you configure the prompt (or sequence of prompts) using frame indices for the duration of the animation sequence.

These Schedule nodes are part of [FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) and the string format follows the [documentation from the same github repo](https://github.com/FizzleDorf/ComfyUI_FizzNodes/wiki/Prompt-Schedules)

---

## Step 4. Configure 2D Motion

![image.png](attachment:77298fa0-c146-491b-80b3-72274300f20e:image.png)

You can modify the values in the **Transform Image** group above to create dramatic effects in the animation, such as infinite zoom or horizontal and vertical traveling.

Edit the **Scale** float value to influence zoom speed.

Edit the **X and Y** float values to influence horizontal and vertical movement speed respectively.

---

## Step 5. (Optional) Enable and Configure Value Schedules

![image.png](attachment:09740b4d-dd8f-4681-9c39-cc8d4ff7ae4b:image.png)

The bypassed groups below the Input and Schedules groups can be used to vary the Denoise, Step count, Seed value, and CFG values over the duration of the animation.

Each group has a "Fast Bypasser" node which you can click on to un-bypass the group, or you can use the "Fast Groups Bypasser" node above to enable/disable them.

Each Schedule group has a Value Schedule node (part of FizzNodes) which can be used to change values over time based on a frame index and number.

---

## Step 6. RUN the workflow

Your final generation result will be visible in the Outputs group below:

You can change the frame_rate and filename_prefix as desired.

![image.png](attachment:fe91c10d-b2b3-44da-9c0d-1ac215cd4c5f:image.png)

---

## Step 7. Profit

---

## Extras:

### Sampling:

![image.png](attachment:e0e40c3a-df0d-4f94-a229-d30d92d46d34:image.png)

This is the group where the FLUX sampler lives. Inputs to the group can be seen on the left as an array of GET nodes, and outputs can be seen on the right as SET nodes.

Generally the “final” values are fed into the sampler, and are set in the “Utils” group seen below.

### Generated Seed List:

![image.png](attachment:ab30fed1-2e06-48c1-a821-fb6c011a6020:image.png)

This group is where this list of seeds for generating noise each frame will be generated.

You can change the “control_mode” to **random** instead of **increment** to get a list of random seed values (including your initial seed set vai the Inputs group).

### Open Loop:

![image.png](attachment:a092d0cf-231d-4521-8450-57c6a654b1df:image.png)

This is the group containing the opening For Loop node, and is responsible for providing the current loop index, a boolean if the current frame is the “start” index (first frame), and additional values passed via the “values” slots 1-4. 

These values exactly correspond to the For Loop Close group’s value slots 1-4. Any value (can be any type) passed to the For Loop Close value 1 (for example) will be present in the next iteration of the loop in For Loop Open value 1, and so on.

### Close Loop:

![image.png](attachment:e5483db0-b7f5-4444-84d4-5ab4c51af091:image.png)

Then end point of the For Loop for each iteration. Values passed to “initial_values” 1-4 will be present in the next iteration in For Loop Open’s values 1-4.
Once the loop is complete, any remaining values in the “initial_values” slots will pass to the output value slots, and execution will continue.

---

## Utility Groups:

### Fast Groups Bypasser:

![image.png](attachment:8a2c71a2-e700-42cb-9acb-8b5efdd6b40e:image.png)

Use the Fast Groups Bypasser node to selectively bypass groups remotely

### Initial Denoise

![image.png](attachment:fbf78f95-7fd7-4af7-8a9f-62fa5bd0d466:image.png)

This utility group will set the initial denoise value to 1.0 (regardless of the default denoise or denoise schedule values) if an upload image is NOT used. This is to ensure the first frame is of high quality for subsequent generation frames.

### Util Switches

![image.png](attachment:14169519-647c-4530-889b-3bc71827d53c:image.png)

The set of switches in the Utils group above are used as a buffer between nodes and groups that transform values (like seed values, latent values, conditioning, etc.) and the final connection to the Sampling group. This is useful if you want to be able to bypass group nodes, and have “fallback” values in place (i.e. use a default seed value if we mute the seed scheduler group instead of crashing)

The any_01 slot is first priority in the switch, followed by any_02, any_03, etc.

For example, if the value in any_01 is found to be None or Null (bypassed), then control will pass to any_02 as a fallback. If any_02 is also invalid then control passes to any_03. And so on.