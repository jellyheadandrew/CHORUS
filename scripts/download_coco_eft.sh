# Download COCO Val 2014
mkdir imports/COCO
mkdir imports/COCO/images
mkdir imports/COCO/annotations
cd imports/COCO/images
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip
cd ../annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
mv annotations/instances_val2014.json instances_val2014.json
rm -r annotations
rm annotations_trainval2014.zip
cd ../../..

# Download COCO-EFT Val 2014 Extension
git clone https://github.com/facebookresearch/eft.git imports/eft
cd imports/eft
sh scripts/download_eft.sh
cd ../..