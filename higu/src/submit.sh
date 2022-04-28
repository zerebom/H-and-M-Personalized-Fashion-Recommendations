COMPETITON=h-and-m-personalized-fashion-recommendations
FILE_PATH=/home/ec2-user/h_and_m/output/log/exp10/submission_df.csv
kaggle competitions submit -c $COMPETITON -f $FILE_PATH -m "validリーク、testの候補生成top300, ルールベースN30, 前回購入n_weeks8"