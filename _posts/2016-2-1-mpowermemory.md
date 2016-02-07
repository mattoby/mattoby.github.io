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




##  What can a memory test tell us about Parkinson's?

Memory is affected in Parkinson's, but usually in late stages of the disease. The more common symptoms, which affect nearly all Parkinson's patients, all involve degradation of motor abilities. Motor symptoms of Parkinson's include tremors, slowness of movement, rigidity, and instability in walking and balancing. Since these symptoms are more ubiquitious among patients than memory issues, it is unclear if a test of memory will be informative. Fortunately, the records from the memory game include data that might give a hint into motor issues aside from memory issues, with a bit of clever feature engineering. 

##  A peek at the data

The mPower app is available for anybody to download and use, and to use as often as they wish. As a result, unlike in a controlled clinical trial, there are no restrictions on the number of times a user may play the memory game. 


![numrecords_park_nopark_2.png]({{site.baseurl}}/images/numrecords_park_nopark_2.png)



As the plots above show, many users played the memory game only once or twice, while a few users played it dozens or hundreds of times. This uneven distribution led to many challenges in data analysis that had to be overcome to make meaningful predictions.

![inthedata1.png]({{site.baseurl}}/images/inthedata1.png)

The first thing I did is looked at how the memory game score correlates with users having Parkinson's. It turned out, this score, taken alone, is not very informative.

![gamescorehist.png]({{site.baseurl}}/images/gamescorehist.png)

## Balancing age

![agehist3.png]({{site.baseurl}}/images/agehist3.png)


##  Feature engineering, i.e., squeezing juice from the app

The game outputs a 'game score', which is intended to assess memory. In the raw records from gameplays, I had access to the regions considered 'correct' to touch for each flower, the order in which the flowers lit up, and the location and time of each touch by the user. I modeled these data as shown in the figure, calculating from these raw data the distance between each 'successful' touch and the center of the flower. This 'touch distance' might indicate an inability of users to hold their hands steady. I also extracted the timing of touches, which I split into two types of features. First, I tracked the time before first touch in each game, i.e., the latency. Next, I averaged the time between each pair of touches after the first one, for a mean touch delay. I aggregated these features separately for plays of the 2x2 game, the 3x3 game, and the 4x4 game, since they differ considerably in difficulty. These features, along with the game score, formed my feature set for predicting the health status of people who played the memory game.

![featureengineering1.png]({{site.baseurl}}/images/featureengineering1.png)

##  Predicting Parkinson's with a random forest model

![model1_roc.png]({{site.baseurl}}/images/model1_roc.png)

![model1_stats.png]({{site.baseurl}}/images/model1_stats.png)

![model1_featureimportances.png]({{site.baseurl}}/images/model1_featureimportances.png)

##  Challenges with uncontrolled sampling

## For other researchers

You can see all the code I used to do this analysis here:

The data from the memory test is held in Synapse, Sage's web portal for data. Anyone can access it if they sign up for a Synapse account and go through a certification procedure. Once a user is certified, they can access the data from the memory test here: [mPowerSynapse](https://www.synapse.org/#!Synapse:syn4993293/wiki/ "mPower study data")



Partnered with ![sage_logo.jpg]({{site.baseurl}}/images/sage_logo.jpg) and ![mpowerparkinsons.svg]({{site.baseurl}}/images/mpowerparkinsons.svg) mPower
