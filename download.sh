# downloading and preparing the five datasets

cd ../omnidata
unzip urban_sunny.zip && rm urban_sunny.zip
unzip urban_cloudy.zip && rm urban_cloudy.zip
unzip urban_sunset.zip && rm urban_sunset.zip
unzip urban_omnidepth_gt_640.zip && rm urban_omnidepth_gt_640.zip
cp -r omnidepth_gt_640 ./sunny/ && cp -r omnidepth_gt_640 ./cloudy/ && mv omnidepth_gt_640 ./sunset/
cp urban_config.yaml ./sunny/config.yaml && cp urban_config.yaml ./cloudy/config.yaml && cp urban_config.yaml ./sunset/config.yaml 
rm urban_config.yaml


unzip omnihouse.zip && rm omnihouse.zip
unzip house_omnidepth_gt_640.zip && rm house_omnidepth_gt_640.zip
mv omnidepth_gt_640 ./omnihouse/
mv house_config.yaml ./omnihouse/config.yaml


unzip omnithings.zip && rm omnithings.zip
unzip things_omnidepth_gt_640.zip && rm things_omnidepth_gt_640.zip
mv omnidepth_gt_640 ./omnithings/
wget -c http://cvlab.hanyang.ac.kr/project/omnistereo/data/things_config.yaml
mv things_config.yaml ./omnithings/config.yaml

mv ./omnithings/cam2/0001.png omnithings/cam2/00001.png
mv ./omnithings/cam2/0002.png omnithings/cam2/00002.png
mv ./omnithings/cam2/0003.png omnithings/cam2/00003.png

mv ./omnithings/cam3/0001.png omnithings/cam3/00001.png
mv ./omnithings/cam3/0002.png omnithings/cam3/00002.png
mv ./omnithings/cam3/0003.png omnithings/cam3/00003.png

mv ./omnithings/cam4/0001.png omnithings/cam4/00001.png
mv ./omnithings/cam4/0002.png omnithings/cam4/00002.png
mv ./omnithings/cam4/0003.png omnithings/cam4/00003.png


# getting real samples
git clone https://github.com/hyu-cvlab/omnimvs-pytorch.git
mv omnimvs-pytorch/data/itbt_sample ./
rm -rf omnimvs-pytorch

git clone https://github.com/hyu-cvlab/sweepnet.git
mv sweepnet/real_indoor_sample ./
rm -rf sweepnet






