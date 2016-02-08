---
layout: post
title: "Parkinson's disease tracking through mobile app memory test"
published: true
---














Healthcare is changing rapidly. The old model of tracking patients through sporadic doctor's visits is becoming antiquated with the enormous innovations in social media, mobile phone technology, and integration of data that have transformed so many fields. These platforms are only now being adopted for tracking patient wellness, but are rapidly meeting the expectations through initiatives such as the Apple ResearchKit, which was published last year. 

Using the new [Apple ResearchKit](http://www.apple.com/researchkit/ "iphone researchkit") technology, Sage Bionetworks has initiated a project called [mPower](http://parkinsonmpower.org/ "Mpower Parkinson's Site"), intended to produce apps to track different diseases, in the hopes of overhauling the model of healthcare tracking and move it to a more dynamic system commesurate with today's hi-tech world. 

In consultation with Sage, I have done an [Insight Data Science](http://insightdatascience.com/ "Insight Data Science") project analyzing data from this mobile app. This analysis is the focus of this blog.

##  The mPower memory test

The mPower mobile app was designed by Sage Bionetworks, using the apple ResearchKit, to track disease progress in Parkinson's patients. Patients download the app from the apple istore and opt in for the clinical trial. Then, they are prompted occasionally to complete tasks that track how they are doing. These tasks include a walking gait test, a vocal test, and others. Data from the tasks are collected and stored in a database intended for use by researchers, and become part of a broader mission to improve disease tracking in Parkinson's between doctor visits. In the future, if a patient's medicine is not working, the app may signal them that they should see their doctor. 

One of the tests within the mPower app is a memory game, which intends to track how the memory of Parkinson's patients varies on a day-to-day basis. The memory app looks like this:

![memorygame1.png]({{site.baseurl}}/images/memorygame1.png)

First, the user is prompted about whether they recently took their Parkinson's medication. Patients can thus be classed into those at their 'best' (just took meds), vs. those at their 'worst' (immediately before taking medication). Then, a 3x3 grid of flowers appears on the screen, and the flowers light up in a randomized order. FInally, the user is tasked with touching the flowers in the same order they lit up. If they do poorly, the app reduces the number of flowers to a 2x2 grid. If they do well, then the complexity increases to a 4x4 grid. If they continue to do well, the number of flowers they have to remember that lit up also increases. 

The game gets quite hard when it gets into 4x4 mode, and you have to track 3, then 4, then 5 flowers. But don't take my word for it -- if you have an iphone, [try it yourself](https://itunes.apple.com/us/app/parkinson-mpower-study-app/id972191200?mt=8 "mpower on itunes"). It will be for a good cause. non-Parkinson's patients are important in this study too, as they provide a baseline for comparison. 

##  What can a memory test say about Parkinson's?

Memory is affected in Parkinson's, but usually in late stages of the disease. The more common symptoms, which affect nearly all Parkinson's patients, all involve degradation of motor abilities. Motor symptoms of Parkinson's include tremors, slowness of movement, rigidity, and instability in walking and balancing. Since these symptoms are more ubiquitious among patients than memory issues, it is unclear if a test of memory will be informative. Therefore, my tasks in this project were twofold:

---

1. Build a predictor of which patients have Parkinson's.

2. Determine which features are the most predictive of disease state.

---

These goals would set a solid basis for future analyses to be carried out by other researchers.

##  A peek at the data

The mPower app is available for anybody to download and use, and to use as often as they wish. As a result, unlike in a controlled clinical trial, there are no restrictions on the number of times a user may play the memory game. Therefore, many users played the memory game only once or twice, while a few users played it dozens or hundreds of times. These highly uneven distributions can be seen in the histograms below:


![numrecords_park_nopark_withrug.png]({{site.baseurl}}/images/numrecords_park_nopark_withrug.png)


It was important to account for these distributions in some way in order to get meaningful results from the data. For example, if I treated each game record as equal in my analyses, a single Parkinson's patient who played the memory game 300 times would have the same weight as 300 individuals who each played the game once. This discrepancy could strongly bias my results in unproductive ways. I handled this distribution issue by using only a single reading per user - either a single random record, or the average for that user over every time they played the memory game. 

The next issue I had to contend with was a large difference in the distribution of ages between Parkinson's and non-Parkinson's patients. Below the age of 45, there were practically no Parkinson's patients in the cohort. While it would be nice to extend my analysis of Parkinson's to young users of the app, it was unrealistic given the data. Therefore, I put a hard cutoff of ages, and only included users 45 years old or older in my analyses.

![agehists.png]({{site.baseurl}}/images/agehists.png)

Beyond this rough age matching, I toyed around with resampling the non-Parkinson's patients to better match the distribution of ages of the Parkinson's patients. However, I did not proceed with this approach, as it required me cutting out more users than I was willing to for the sake of clean data. When many more users have played the memory game, I believe that better age matching using this sort of a resampling approach may become prudent. 

##  Digging into the features

From each user session, the memory game tracks an overall 'memory score' (denoting accuracy in touching the flowers), as well as a detailed record of screen taps during gameplay. There is also a record of whether the user is currently on medication, as self reported. Aside from this, I had available each user's disease status, as well as some demographic information.

![mpower_data_overview.png]({{site.baseurl}}/images/mpower_data_overview.png)

The first thing I did is looked at how the memory game score correlates with users having Parkinson's. It turned out, this score, taken alone, is not tremendously informative about disease status.

![memoryscorehist.png]({{site.baseurl}}/images/memoryscorehist.png)

Therefore, I turned to the game records themselves. In these raw records, I had access to the regions considered correct to touch for each flower, the order in which the flowers lit up, and the location and time of each touch by the user.

As I mentioned before, Parkinson's is primarily a disease of the motor system. Therefore, I hypothesized that there could be two new groups of features, aside from memory-based features like the game score, which might be informative:

1. Features having to do with time delay between touches, which could indicate _Bradykinesia_ (slowness of movement) or _Akinesia_ (difficulty initiating movements). 

2. Features having to do with the distance between the intended targets and actual screen taps, which could indicate _Dyskinesia_ (difficulty of controlling movement, e.g., with tremor). 

With this in mind, I calculated from the raw data the distance between each 'successful' touch and the center of the intended flower, as well as the time delays between touches and also before the first touch.


![feature_engineering.png]({{site.baseurl}}/images/feature_engineering.png)


I split the touch timings into two types of features. First, I tracked the 'reaction time', i.e., the time before first touch in each game. Next, I averaged the time between each pair of touches after the first one, to get an averaged time between touches. I aggregated each of these features separately for plays of the 2x2 game, the 3x3 game, and the 4x4 game, since the games differ considerably in difficulty. These features, along with the game score, formed my feature set for predicting the health status of people who played the memory game.

##  Predicting Parkinson's

I approached the task of classifying users into Parkinson's and non-Parkinson's groups using a logistic regression model. I found that I achieved the best results with little or no regularization, which I speculate is due to a broad spred of information among the features (beyond the few top features that hold most of the information, as seen below). I split my data 70%/30% into training and test sets, and then performed machine learning. 

As seen in the Receiver Operating Characteristic curves below, my model is able to predict whether users of the memory game have Parkinson's with an area under the ROC curve of 0.74-0.78, depending on the particulars of the analysis. This means that, given game records from a Parkinson's patient and a non-Parkinson's user, the model will give a higher "likelihood of Parkinson's" score to the actual patient 74-78% of the time. A random predictor would, of course, predict at 50%. 

![ROCs_allfeatures.png]({{site.baseurl}}/images/ROCs_allfeatures.png)

Logistic regression calculates coefficients for each feature, which can be interpreted as relative importances of the features for doing the classification (i.e., for predicting whether a game user has Parkinson's). Looking at these coefficients can give a lot of insight into what a model is doing. 

![feature_importances_allfeatures.png]({{site.baseurl}}/images/feature_importances_allfeatures.png)

There are two striking observations to be made from the distribution of these coefficients: 

1. The most informative features by far are the mean times between taps (especially in the 3x3 game, but also in the 4x4 game). I found that removing these features severely reduces predictiveness of the model.

2. The memory score very little predictive power. It is nearly last in the ranking of feature importances (note, negative coefficients denote importance as well -- so memory score, having a coefficient near zero, is one of the least informative features in the list). I found that the model's predictiveness is not affected by removal of the memory score.

I found these observations very interesting. To follow up, I took a look at the distribution of Parkinson's and non-Parkinson's scores on some of the most informative features that came out of my analysis. Unlike the memory score, it is clear that the timing between taps say something about whether a user has Parkinson's.

![distance_and_dt_dists.png]({{site.baseurl}}/images/distance_and_dt_dists.png)


I was surprised, on the other hand, that the distribution of reaction times (i.e., the length of time before the first tap in a given game) did not vary strongly between Parkinson's from non-Parkinson's patients. However, as can be seen in the plot below, there are a handful of Parkinson's patients with excessively long reaction times, and 


##  Challenges with uncontrolled sampling

I did found, troublingly, that a logistic regression model composed of only the demographic features of age, education level, and gender, and containing no features whatsoever derived from the memory game, predicts whether a user has Parkinson's with an Area under the curve of 0.81 -- even better than my model trained from the memory game. This is obviously not right, and betrays a strong bias in the demographic distribution of the data (see my previous discussion about resampling). To avoid contaminating my models, I excluded all of these demographic features outright. However, given a more demographically balanced dataset (which might exist in the future when many more people have played the memory game), education level and especially age would likely become useful to be added to a model as interaction terms.





## For other researchers

You can see all the code I used to do this analysis here:

[An ipython notebook that summarizes my analyses](https://github.com/mattoby/mpower_memory/blob/master/Memory_summary_analyses.ipynb "overview notebook")


The data from the memory test is held in Synapse, Sage's web portal for data. Anyone can access it if they sign up for a Synapse account and go through a certification procedure. Once a user is certified, they can access the data from the memory test here: [mPowerSynapse](https://www.synapse.org/#!Synapse:syn4993293/wiki/ "mPower study data")


Partnered with ![sage_logo.jpg]({{site.baseurl}}/images/sage_logo.jpg) and ![mpowerparkinsons.svg]({{site.baseurl}}/images/mpowerparkinsons.svg) mPower
