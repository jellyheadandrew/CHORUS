NUM_BATCH_PER_AUGPROMPT=${1:-20}
BATCH_SIZE=${2:-6}

cp prompts/demo/surfboard.txt prompts/demo/surfboard-demo.txt
python scripts/generation/generate_images.py \
--skip_done \
--num_batch_per_augprompt $NUM_BATCH_PER_AUGPROMPT \
--batch_size $BATCH_SIZE \
--categories 'surfboard-demo'
rm prompts/demo/surfboard-demo.txt