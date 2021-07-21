# Video-Surveillance-system
* A system to automate the task of analyzing video surveillance 
* Analyze the video feed in real-time and identify abnormal activity. 
### Introduction:
•	Events that are of interest in long video sequences, such as surveillance footage, often have an extremely low probability of occurring. As such, 
  manually detecting such events, or anomalies, is a very fussy job that often requires more manpower than is generally available.
  
•	There is an increasing need not only for recognition of objects and their behavior, but for detecting the rare, interesting occurrences of unusual 
  objects or suspicious behavior in the large body of ordinary data.
  
•	This has prompted the need for automated detection and segmentation of sequences of interest.

•	Video data is challenging to represent and model due to its high dimension- ality, noise, and a huge variety of events and interactions.

•	Early events are based on some predefined heuristics, which makes the detection model difficult to generalize to different surveillance scenes.

•	Our proposed method is domain free (i.e., not related to any specific task, no domain expert required), does not require any additional human effort, 
  and can be easily applied to different scenes.
  
•	To prove the effectiveness of the proposed method we apply the method to real-world datasets and show that our method consistently outperforms similar 
  methods while maintaining a short running time.
  
•	By incorporating convolutional feature extractor in both spatial and temporal space into the encoding-decoding structure, we build an end-to-end 
  trainable model for video anomaly detection. The advantage of our model is that it is semi-supervised – the only ingredient required is a long video 
  segment containing only normal events in a fixed view. Despite the model’s ability to detect abnormal events and its robustness to noise, depending on 
  the activity complexity in the scene, it may produce more false alarms compared to other methods.
  
<c>![image](https://user-images.githubusercontent.com/70462853/125668521-51f27703-153f-426a-94dc-f21ed8554c5c.png)</c>

### Algorithm:
1.	Import Directories
2.	Initialize directory path variable and describe a function to process and store video frames
3.	Extract frames from video and call store function
4.	Feature Extraction
5.	Create spatial autoencoder architecture
6.	Training and testing
7.	Detection abnormality
8.	Visualization


### Block Diagram
![image](https://user-images.githubusercontent.com/70462853/125673456-870f0df2-90b1-4f4a-8d87-5101a4537e5c.png)
![image](https://user-images.githubusercontent.com/70462853/125673500-b038aaad-290c-401b-92ec-65e89ac598f8.png)

### Software Requirements:
•	Python IDE, e.g., jupyter notebook, colab

•	Keras-Snippet: -

This is a keras code snippet extension that provide boilerplate for different CNN algorithm and as well as for various element involved at the time of writing deep learning code from scratch. This snippet reduces the coding time
•	TensorFlow Snippets: -

TensorFlow is a free and open-source software library for machine learning. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.

### Debugging:

	Extracting frames and giving them as input would result in underfitting problem. Therefore, we need to convert them in greyscale, resize and normalize and then clip values between 0-1.

	Early stopping and checkpoints are set default to save best model on min method, therefore we can try saving on mode=’min’, ‘max’ as per requirement on which monitor we are evaluating.

	Training error: ‘val_loss’, ‘val_acc’, this error can be solved by defining validation_split in model.fit(). Which split training data into training and validation data for proper training and validation.

	Data processing and Training time can be calculated by using start_time as datetime.now() and final time as datetime.now()-start_time


### Complexities Involved:
•	Technical:
-	Video storage
-	Network Utilization
-	Computational Needs
-	Video Verification
•	Privacy and Security
-	Security in VSaaS System
-	Privacy issue arises from new analytics
-	Privacy in crowdsourcing solutions

### Few processing output:
1. frame reading after extrtaction 

![image](https://user-images.githubusercontent.com/70462853/125675241-05d086ed-cc07-484d-a13b-e5cd8e1dbbe4.png)

2. Data processing

![image](https://user-images.githubusercontent.com/70462853/125675303-6cebb1b2-9414-471b-aae8-3ffa503a0ad1.png)

3. CNN model/architecture

![image](https://user-images.githubusercontent.com/70462853/125675375-5886e814-cff9-474f-ae65-d82f11eab597.png)

4. Training

![image](https://user-images.githubusercontent.com/70462853/125675410-4cded11c-aca6-4b18-851f-19ad793189df.png)

5. Output

![image](https://user-images.githubusercontent.com/70462853/125675461-de7e281a-7585-45eb-8ad0-d4e1182a4c0e.png)

### Result analysis:

1.)	Input frames are resized to 227x227x10 so that all input images are of same size leading to better training

2.)	Our CNN model consists of 1 input layer. 7 hidden layer, 1 Output layer

  1st hidden layer (conv3D): No of training parameter=15616 with dimensions (55, 55, 10, 128)

  2nd hidden layer(conv3D): No of trainable parameter=204864 with dimensions (26, 26, 10, 64)

  3rd hidden layer(convLSTM2D): No of trainable parameter=296168 with dimensions (26, 26, 10, 64)

  4th hidden layer(convLSTM2D): No of trainable parameter=110720 with dimensions (26, 26, 10, 32)

  5th hidden layer(convLSTM2D): No of trainable parameter=221440 with dimensions (26, 26, 10, 64)

  6th hidden layer(conv3DTran): No of trainable parameter=204928 with dimensions (55, 55, 10, 128)

  7th hidden layer(conv3DTr): No of trainable parameter=15489 with dimensions (227, 227, 10, 1)

  Total trainable parameter = 1068225 and non-trainable parameter = 0

3.)	Training on fps=1 (i.e. total frame=613); 
  Loss decreased from 0.0888 to 0.0697

  Accuracy increased on 52.06% to 54.26%

  Reconstruction loss for proper detection>0.0003

  Training on fps=5 (i.e. = 15251); 

  loss decreased from 0.0888 to 0.0212

  Accuracy increased from 66.23% to 77.99%   

  val_loss decreased from 0.0920 to 0.0476

  val_acc increased from 71.77% to 76.10%		

  Reconstruction loss for proper detection>0.00042

4.)	Using early-stopping we get epoch required for proper training is 10 when monitored on “val_loss”, with accuracy= 78.01%, loss= 0.0212, val_loss= 0.476, val_acc= 76.10%
  and patience =3

5.)	Model checkpoint monitored on “mse” saves best model only and replaces earlier one as better model is achieved.

6.)	Model is best achieved keeping batch_size=1 (i.e. training one frame at one time)


7.)	Time taken for data processing is 00:01:52:4682 hrs

  Time taken for training model =05:57:20:119 hrs
8.)	Model is best achieved with

  Optimizer=’adam’, 

  loss=’mse’

  Metrics=’accuracy’,

  Activation function used = tanh

![image](https://user-images.githubusercontent.com/70462853/125676112-3b79a637-579d-4544-9f71-c77e707fde91.png)

![image](https://user-images.githubusercontent.com/70462853/125676138-e561808f-85b4-42cf-9998-cbd418bb6642.png)

### Conclusion:

We formulate anomaly detection as a spatiotemporal sequence outlier detection problem and applied a combination of spatial feature extractor and temporal sequencer ConvLSTM to tackle the problem. The ConvLSTM layer not only preserves the advantages of FC-LSTM but is also suitable for spatiotemporal data due to its inherent convolutional structure. By incorporating convolutional feature extractor in both spatial and temporal space into the encoding-decoding structure, we build an end-to-end trainable model for video anomaly detection. The advantage of our model is that it is semi-supervised – the only ingredient required is a long video segment containing only normal events in a fixed view. Despite the model’s ability to detect abnormal events and its robustness to noise, depending on the activity complexity in the scene, it may produce more false alarms compared to other methods. For future work, we will investigate how to improve the result of video anomaly detection by active learning – having human feedback to update the learned model for better detection and reduced false alarms. One idea is to add a supervised module to the current system, which the supervised module works only on the video segments filtered by our proposed method, then train a discriminative model to classify anomalies when enough video data has been acquired.

### References:

[1]	Adam, A., Rivlin, E., Shimshoni, I., Reinitz, D.: Robust real-time unusual event detection using multiple fixed-location monitors. IEEE Transactions on Pattern Analysis and Machine Intelligence 30(3), 555–560 (2008)

[2]	Kratz, L., Nishino, K.: Anomaly detection in extremely crowded scenes using spatio-temporal motion pattern models. In: 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2009. pp. 1446–1453 (2009)

[3]	Mahadevan, V., Li, W., Bhalodia, V., Vasconcelos, N.: Anomaly detection in crowded scenes. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) pp. 1975–1981 (2010)

[4]	Cong, Y., Yuan, J., Liu, J.: Sparse reconstruction cost for abnormal event detection. In: Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition. pp. 3449–3456 (2011)

[5]	Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., Fei-Fei, L.: Large-scale video classification with convolutional neural networks. In: 2014 IEEE Conference on Computer Vision and Pattern Recognition. pp. 1725–1732 (June 2014)

[6]	Du Tran1, Lubomir Bourdev,  Rob Fergus , Lorenzo Torresani , Manohar Paluri.: Learning Spatiotemporal Features with 3D Convolutional Networks, 2015

[7]	Hasan, M., Choi, J., Neumann, J., Roy-Chowdhury, A.K., Davis, L.S.: Learning temporal regularity in video sequences. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 733–742 (June 2016)

[8]	Virgil Claudiu Banu, Ilona Mădălina Costea, Florin Codrut Nemtanu and Iulian Bădescu.: Intelligent Video Surveillance System. IEEE 23rd International Symposium for Design and Technology in Electronic Packaging (SIITME) 2017.

[9]	Yong Shean Chong, Yong Haur Tay.: Abnormal Event Detection in Videos using Spatiotemporal Autoencoder. Lee Kong Chian Faculty of Engineering Science, Universiti Tunku Abdul Rahman, 43000 Kajang, Malaysia.2017

[10]	. C. Banu, I. M. Costea, F. C. Nemtanu and I. Bădescu, "Intelligent video surveillance system," 2017 IEEE 23rd International Symposium for Design and Technology in Electronic Packaging (SIITME), 2017, pp. 208-212, doi: 10.1109/SIITME.2017.8259891

[11]	G. F. Shidik, E. Noersasongko, A. Nugraha, P. N. Andono, J. Jumanto and E. J. Kusuma, "A Systematic Review of Intelligence Video Surveillance: Trends, Techniques, Frameworks, and Datasets," in IEEE Access, vol. 7, pp. 170457-170473, 2019, doi: 10.1109/ACCESS.2019.2955387.


