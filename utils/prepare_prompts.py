import os
from glob import glob


def prepare_cps(prompt_dir, categories, use_txt_prompts: bool):
    if categories is None and use_txt_prompts:
        categories = [
            prompt_pth.split("/")[-1].replace(".txt", "")
            for prompt_pth in sorted(list(glob(f"{prompt_dir}/*.txt")))
        ]
    elif categories is None and not use_txt_prompts:
        categories = [
            prompt_dir.split("/")[-1]
            for prompt_dir in sorted(list(glob(f"{prompt_dir}/*")))
        ]
    else:
        pass

    # final cps to return
    final_cps = []

    # load category-prompt pairs
    for category in categories:
        if not use_txt_prompts:
            cps = [tuple(prompt_pth.split("/")[-2:]) 
                    for prompt_pth 
                    in list(glob(f"{prompt_dir}/{category}/*"))]
        else:
            txt_pth = f"{prompt_dir}/{category}.txt"
            with open(txt_pth, "r") as f:
                cps = [
                        tuple(
                            [os.path.splitext(os.path.basename(txt_pth))[0], 
                            line.strip()]
                        ) for line in f.read().split("\n")
                    ]

        # remove empty string
        cps = [tuple([category, prompt]) 
                    for category, prompt in cps
                    if len(prompt) > 0
                ]
        # remove prompt starting with #
        cps = [tuple([category, prompt]) 
                    for category, prompt in cps
                    if prompt[0] != "#"
                ]
        # remove . in end
        cps = [tuple([category, prompt[:-1]]) if prompt[-1] == "." 
                    else tuple([category, prompt]) 
                    for category, prompt in cps]
        # remove duplicates
        cps = sorted(list(set(cps)))
        # append to final_cps
        final_cps += cps
 
    return final_cps


def prepare_augdict(cps, augmentation_path, verbose):
    with open(augmentation_path, "r") as f:
        # read augmentations
        augprompts = [line.strip() for line in f.read().split("\n") if line.strip() != ""]
        augprompts = [line.strip() for line in augprompts if line.strip()[0] != "#"]
        # header starts with "AUGPROMPT_COMMON": use augmentations for all prompts
        if augprompts[0] == "AUGPROMPT_COMMON":
            if verbose:
                print("Using Common Augprompts over All Prompts...")
            augprompts = augprompts[1:]
            tmp = dict()
            for cp in cps:
                tmp[cp] = augprompts
            augdict = tmp
        # header starts with "AUGPROMPT_PER_PROMPT": augmentations specified for each prompt     
        elif augprompts[0] == "AUGPROMPT_PER_PROMPT":
            if verbose:
                print("Using Prompt-specific Augprompts...")
            augprompts = augprompts[1:]
            tmp = dict()
            current_category = ""
            current_prompt = ""
            for line in augprompts:
                if line[:12] == "category:":
                    current_category = line.replace("category:", "").strip()
                elif line[:7] == "PROMPT:":
                    current_prompt = line.replace("PROMPT:", "").strip()
                    assert (current_category, current_prompt) not in tmp.keys(), f"Duplicate prompt in {augmentation_path} file"
                    assert (current_category, current_prompt) in cps, f"In {augmentation_path}, there are invalid prompts."
                    tmp[(current_category, current_prompt)] = []
                elif current_category == "" or current_prompt == "":
                    continue
                else:
                    tmp[(current_category, current_prompt)].append(line.strip())
            augdict = tmp

        # invalid header => raise error    
        else:
            assert False, "Augprompt-header must be specified."
    
    # sort augdict
    augdict = dict(sorted(augdict.items()))
    assert tuple(sorted(list(augdict.keys()))) == tuple(sorted(cps)), f"Mismatch in augprompt file: '{augmentation_path}' & cps we are to use: {cps}"

    return augdict


def prepare_cps_from_dirs(args, image_dir, use_filter_setting=False):
    # Categories. If None, type them in.
    if args.categories is None:
        categories = ["*"] # all
    else:
        assert type(args.categories) == list
        categories = args.categories

    # Parse Categories.
    if type(categories) == str:
        categories = [category.strip() for category in categories.split(",")]
    else:
        assert type(categories) == list

    # To Save cps.
    cps = []

    # If using setting (usually After Filtering)
    if use_filter_setting:
        for category in categories:
            cps += [
                # modified for "settings" input
                tuple([prompt_pth.split("/")[-3],prompt_pth.split("/")[-1]]) 
                for prompt_pth in list(glob(f"{image_dir}/*/{category}/settings:*/*"))
            ]
    # If not using setting (usually Before Filtering)
    else:
        for category in categories:
            cps += [
                tuple(prompt_pth.split("/")[-2:]) 
                for prompt_pth in list(glob(f"{image_dir}/*/{category}/*"))
            ]

    # Remove duplicate, and sort
    cps = sorted(list(set(cps)))
    
    # Filter the prompts.
    if 'prompts' not in vars(args).keys():
        pass
    elif args.prompts is not None:
        _cps = []
        for prompt_type, prompt in cps:
            if prompt in args.prompts:
                _cps += [(prompt_type, prompt)]
        cps = _cps
        # print that filtering with 'args.prompts' has happend
        if args.verbose:
            print(f"Additionally filtered category-prompt pairs sicne '--prompts' is provided: Using {len(cps)} number of cps\n")

    # print total number of (category, prompt) pairs
    if args.verbose:
        print(f"Running for {len(cps)} number of prompts\n\n")    

    return cps


def get_unique_categories(cps):
    # list to save all categories
    all_categories = []

    # iterate for all category-prompt pairs
    for category_, prompt_ in cps: 
        all_categories.append(category_)

    # remove duplicates to retrieve unique categories
    unique_categories = sorted(list(set(all_categories)))

    return unique_categories


def get_cps_for_given_category(cps, category):
    # list to save prompts for given category 
    filtered_prompts = []

    # iterate for all category-prompt pairs
    for category_, prompt_ in cps:
        if category_ == category:
            filtered_prompts.append((category_, prompt_))
    
    # remove duplicates to retrieve cps for given category
    filtered_prompts = sorted(list(set(filtered_prompts)))
    return filtered_prompts