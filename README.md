# Machine vs bot: tuning Machine Learning models to detect bots on Twitter
Code for the WCNPS paper "Machine vs Bot: tuning machine learning models to detect bots on Twitter" 
submited to the Workshop on Communication Networks and Power Systems (WCNPS) 2020.

    Abstract — Bot generated content on social media canspread fake news and hate speech, manipulate 
    public opinion and influence the community on relevant topics, such as elections. Thus, bot 
    detection in social media platforms plays an important role for the health of the platforms and
    for the well-being of societies. In this work, we approach the detection of bots on Twitter as a
    binary output problem through the analysis of account features. We propose a pipeline for feature 
    engineering and model training, tuning and selection. We test ourpipeline using 3 publicly 
    available bot datasets, comparing the performance of all trained models with the model selected 
    at the end of our pipeline.
    
    Keywords — machine-learning, bot detection

The `data` folder contains three datasets copied from previous works:

1. **cresci-stock**: S. Cresci, F. Lillo, D. Regoli, S. Tardelli, and M. Tesconi, “Cash-tag piggybacking:  
Uncovering spam and bot activity in stockmicroblogs on twitter", ACM Transactions on the Web (TWEB)", vol. 
13, no. 2, pp. 1–27, 2019.
2. **botwiki**: K.-C. Yang, O. Varol, P.-M. Hui, and F. Menczer, “Scalableand generalizable social bot 
detection through data selection”, Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, 
no. 01, p. 1096–1103, Apr 2020.
4. **cresci-rtbust**: M. Mazza, S. Cresci, M. Avvenuti, W. Quattrociocchi, and M. Tesconi, “Rtbust: 
Exploiting temporal patterns for botnet detection on twitter", in Proceedings of the 10th ACM Conference 
on Web Science, 2019, pp. 183–192.