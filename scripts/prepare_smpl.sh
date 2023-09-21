# get smpl
cd imports/frankmocap
mkdir extra_data
cp SMPL_python_v.1.1.0.zip extra_data/SMPL_python_v.1.1.0.zip
cd extra_data
rm -r SMPL_python_v.1.1.0
unzip SMPL_python_v.1.1.0.zip
mkdir smpl
mv SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
rm -r SMPL_python_v.1.1.0
rm SMPL_python_v.1.1.0.zip
cd ../../..