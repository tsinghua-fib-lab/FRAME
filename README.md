# FRAME
The code and datasets of paper ”Learning Fine-grained User Interests for Micro-video Recommendation“ (SIGIR 2023)

Dataset link：[https://drive.google.com/file/d/1podEwklx-9P4l91hoLI7IgX3i6yslArk/view?usp=sharing](https://drive.google.com/file/d/1I2fdnPxCbYaHTVAoDUnOsukagQLyTzZM/view?usp=sharing)
## Dataset description 
The detailed explanations of datasets are as follows:
```
| file name              | content                                                                                                                                                                                                    |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| fast_video_feature     | a folder contains each video’s clip visual embedding(each video is divided to 4 clips)                                                                                                                     |
| test_samples           | a folder contains each user’s interaction records in the test set                                                                                                                                          |
| data_final2.csv        | total user interaction records, which contains user-id, video-id, the user’s playing time, duration time of the video, interaction timestamp and multi-level behaviors including like, follow, and forward |
| fast_train_samples.csv | user interaction records for training                                                                                                                                                                      |
| fast_test_samples.csv  | user interaction records for test                                                                                                                                                                          |
```
Some preprocessed data for code running:
```
| file name                | content                                                 |
|--------------------------|---------------------------------------------------------|
| clip_em.npy              | clip embedding matrix                                   |
| pos_a_normed.npy         | normalized form of positive relation adjacency matrix   |
| neg_a_normed.npy         | normalized form of negative relation adjacency matrix   |
| fast_total_u_id_list.npy | list of all user-id                                     |
| model0.pth               | a pre-trained model                                     |
```
