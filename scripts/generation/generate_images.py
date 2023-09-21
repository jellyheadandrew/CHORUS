import os
import argparse
from omegaconf import OmegaConf

import torch
from torch import autocast
import numpy as np

from PIL import Image

from tqdm import tqdm
from einops import rearrange

from imports.ldm.ldm.util import load_model_from_config
from imports.ldm.ldm.models.diffusion.ddim import DDIMSampler
from imports.ldm.ldm.models.diffusion.plms import PLMSSampler

from constants.ldm import DEFAULT_BATCH_SIZE, DEFAULT_LDM_MODEL_KEY, AUGMENTATION_PROMPT_PTH, MODELS, GetLDMOptions
from constants.metadata import DEFAULT_PROMPT_DIR, DEFAULT_IMAGE_DIR, DEFAULT_SEED

from utils.prepare_prompts import prepare_cps, prepare_augdict
from utils.reproducibility import seed_everything




def gen_images(
        cps, 
        augdict, 
        ldm_model_key,
        batch_size, 
        num_batch_per_augprompt,
        augprompts_not_to_use,
        skip_done,
        verbose,
        save_dir,
    ):

    # Create result directory
    result_dir=f"./{save_dir}/{ldm_model_key}"
    os.makedirs(result_dir, exist_ok=True)

    # get options
    opt = GetLDMOptions(model_key=ldm_model_key, models=MODELS)   
    
    # load config & model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", verbose)
    
    # send model to device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    # diffusion sampler
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # clear screen if not verbose
    if not verbose: os.system('clear')

    # if opt.fixed_code, we use fixed initial noisy latent for generation
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    # set precision
    precision_scope = autocast

    # check if you can continue
    for augprompt_not_to_use in augprompts_not_to_use:
        for prompttype_prompt in augdict.keys():
            if augprompt_not_to_use in augdict[prompttype_prompt]:
                augdict[prompttype_prompt].remove(augprompt_not_to_use)
    
    # print augprompts we are using
    if verbose:
        print("\n\n\n\nAugprompts we are using: ")
        for prompttype_prompt in augdict:
            print(f"{prompttype_prompt} => {augdict[prompttype_prompt]}")
        print("\n\n\n\n")


    # generate images
    for category, prompt in cps:
        # declar saving directory
        final_savedir = f"{result_dir}/{category}/{prompt}"
        os.makedirs(final_savedir, exist_ok=True)

        # create a batch per prompt+augmentation
        num_batch = len(augdict[(category, prompt)]) * num_batch_per_augprompt

        # skip if exists
        choose2skip = True
        for augprompt in augdict[(category, prompt)]:
            # make save directory IF there's no directory
            os.makedirs(f"{final_savedir}/{augprompt}", exist_ok=True)
            # check to skip or not
            if verbose:
                print("augprompt: ",augprompt)
            if len(os.listdir(os.path.join(final_savedir, augprompt))) != num_batch_per_augprompt * batch_size:
                choose2skip = False
        if choose2skip and skip_done:
            if verbose:
                print("\n\tContinueing: ", prompt)
            continue

        
        # generate images
        base_count = 0
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in tqdm(range(num_batch), desc=f"Generating Images for [Category '{category}' / Prompt '{prompt}']"):
                        # retrieve augmentation
                        augprompt = augdict[(category, prompt)][n // num_batch_per_augprompt]
                        
                        # make a batch of augmented prompts
                        if augprompt == "original":
                            datas = [[prompt] * batch_size]
                        else:
                            datas = [[prompt+augprompt] * batch_size]

                        # continue or not
                        if len(os.listdir(os.path.join(final_savedir, augprompt))) // batch_size > n % num_batch_per_augprompt and skip_done:
                            if verbose:
                                print(f"\tSkipping batch:{n % num_batch_per_augprompt +1} for prompt+augmentation:{datas[0][0]}")
                            base_count += batch_size
                            continue
                        else:
                            if verbose:
                                print(f"\n\nBatch: {n % num_batch_per_augprompt +1} for for prompt+augmentation:{datas[0][0]} --> Generating...\n\n")
                        
                        # generate data batch
                        if verbose: batch_iterable = tqdm(datas, desc="data")
                        else: batch_iterable = datas
                        for data in batch_iterable:
                            if verbose:
                                print(data)
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(data, tuple):
                                data = list(data)
                            c = model.get_learned_conditioning(data)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=batch_size,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code,
                                                            )
                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            # save images
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(final_savedir, augprompt, f"{base_count:06}.png"))
                                base_count += 1




if __name__ == "__main__":
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=str, nargs="+")
    parser.add_argument("--ldm_model_key", type=str, default=DEFAULT_LDM_MODEL_KEY)
    
    parser.add_argument("--disable_txt_prompts", action="store_true")
    parser.add_argument("--augprompts_path", type=str, default=AUGMENTATION_PROMPT_PTH)
    parser.add_argument("--augprompts_not_to_use", nargs="+")

    parser.add_argument("--prompt_dir", type=str, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_IMAGE_DIR)

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_batch_per_augprompt", type=int)

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    

    # prepare prompts
    cps = prepare_cps(args.prompt_dir, args.categories, not args.disable_txt_prompts)

    # prepare prompt augmentations
    augdict = prepare_augdict(cps, args.augprompts_path, args.verbose)

    # seed for reproducible generation
    seed_everything(args.seed)

    # prepare augmentations to skip
    if args.augprompts_not_to_use is None:
        args.augprompts_not_to_use = []

    # print information
    if args.verbose:
        print(f"Running for {len(cps)} number of (category, prompt) pairs\n\n")
        print("\n====================================\n")
        print("\nAugprompts:")
        for prompt in augdict.keys():
            print(f"{prompt} => ")
            for augprompt in augdict[prompt]:
                print("\t",augprompt)
        print("\n====================================\n")

        print(f"You have specified augprompts not to use: {args.augprompts_not_to_use}")
        # arguments
        print("===Arguments===")
        print(args)


    # generate images
    gen_images(
        cps=cps, 
        augdict=augdict, 
        ldm_model_key=args.ldm_model_key,
        batch_size=args.batch_size, 
        num_batch_per_augprompt=args.num_batch_per_augprompt,
        augprompts_not_to_use=args.augprompts_not_to_use,
        skip_done=args.skip_done,
        verbose=args.verbose,
        save_dir=args.save_dir,
    )