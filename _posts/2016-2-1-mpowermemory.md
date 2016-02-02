---
layout: post
title: "Parkinson's disease tracking through mobile app memory test"
published: true
---



Healthcare is changing rapidly. The old model of tracking patients with sporadic doctor's visits is becoming antiquated with the enormous innovations in constant available through the internet, social media, and mobile phone technology, that are driving other fields. These platforms are only now being adopted for tracking patient wellness, but are rapidly meeting the expectations through initiatives such as the Apple care kit, which was published last year. 

Sage Bionetworks has teamed up with Apple in using their [iphone ResearchKit](http://www.apple.com/researchkit/ "iphone researchkit") to in a project called [mPower](http://parkinsonmpower.org/ "Mpower Parkinson's Site"), intended to produce apps to track different diseases, in the hopes of overhauling the model of healthcare tracking and move it to a more dynamic system commesurate with today's hi-tech world. 

In consultation with Sage, I have done an [Insight Data Science](http://insightdatascience.com/ "Insight Data Science") project analyzing data from this mobile app. This analysis is the focus of this blog.

##  The mPower memory test

The mPower mobile app was designed by Sage Bionetworks, using the apple ResearchKit, to track disease progress in Parkinson's patients. Patients download the app from the apple istore and opt in for the clinical trial. Then, they are prompted occasionally to complete tasks that track how they are doing. These tasks include a walking gait test, a vocal test, and others. Data from the tasks are collected and stored in a database intended for use by researchers, and become part of a broader mission to improve disease tracking in Parkinson's between doctor visits. In the future, if a patient's medicine is not working, the app may signal them that they should see their doctor. 

One of the tests within the mPower app is a memory game, which intends to track how the memory of Parkinson's patients varies on a day-to-day basis. The memory app looks like this:

![mpowerappface.png]({{site.baseurl}}/images/mpowerappface.png)

First, the user is prompted about whether they recently took their Parkinson's medication. Patients can thus be classed into those at their 'best' (just took meds), vs. those at their 'worst' (immediately before taking medication). Then, a 3x3 grid of flowers appears on the screen, and the flowers light up in a randomized order. FInally, the user is tasked with touching the flowers in the same order they lit up. If they do poorly, the app reduces the number of flowers to a 2x2 grid. If they do well, then the complexity increases to a 4x4 grid. If they continue to do well, the number of flowers they have to remember that lit up also increases. 

The game gets quite hard when it gets into 4x4 mode, and you have to track 3, then 4, then 5 flowers. But don't take my word for it -- if you have an iphone, [try it yourself](https://itunes.apple.com/us/app/parkinson-mpower-study-app/id972191200?mt=8 "mpower on itunes"). It will be for a good cause. non-Parkinson's patients are important in this study too, as they provide a baseline for comparison. 

##  What can a memory test tell us about Parkinson's?

Memory is affected in Parkinson's, but usually in late stages of the disease. The more common symptoms, which affect nearly all Parkinson's patients, all involve degradation of motor abilities. Motor symptoms of Parkinson's include tremors, slowness of movement, rigidity, and instability in walking and balancing. Since these symptoms are more ubiquitious among patients than memory issues, it is unclear if a test of memory will be informative. Fortunately, the records from the memory game include rom I was suspicious about the ability of the memory per se to test test 'game score', which is designed to The memory test has the potential to track some of these issues , so I suspected that The memory test outputs a 

##  What the data look like


![confusion_randforest_resampled1.png]({{site.baseurl}}/images/confusion_randforest_resampled1.png)
![park_nopark_resample2_agehist.png]({{site.baseurl}}/images/park_nopark_resample2_agehist.png)
![park_nopark_resample1_agehist.png]({{site.baseurl}}/images/park_nopark_resample1_agehist.png)
![park_nopark_scorehist.png]({{site.baseurl}}/images/park_nopark_scorehist.png)
![park_nopark_agehist.png]({{site.baseurl}}/images/park_nopark_agehist.png)
![sage.png]({{site.baseurl}}/images/sage.png)






fin

Partnering with ![sage_logo.jpg]({{site.baseurl}}/images/sage_logo.jpg) and ![mpowerparkinsons.svg]({{site.baseurl}}/images/mpowerparkinsons.svg) mPower





