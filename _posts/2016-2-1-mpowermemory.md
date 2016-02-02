---
layout: post
title: "Parkinson's disease tracking through mobile app memory test"
published: true
---



Healthcare is changing rapidly. The old model of tracking patients with sporadic doctor's visits is becoming antiquated with the enormous innovations in constant available through the internet, social media, and mobile phone technology, that are driving other fields. These platforms are only now being adopted for tracking patient wellness, but are rapidly meeting the expectations through initiatives such as the Apple care kit, which was published last year. 

Sage Bionetworks has teamed up with Apple in a project called [mPower](http://parkinsonmpower.org/ "Mpower Parkinson's Site"), intended to produce apps to track different diseases, in the hopes of overhauling the model of healthcare tracking and move it to a more dynamic system commesurate with today's hi-tech world. 

In consultation with Sage, I have done an [Insight Data Science](http://insightdatascience.com/ "Insight Data Science") project analyzing data from this mobile app. This analysis is the focus of this blog.

##  The mPower memory test

The mPower mobile app was designed by Sage Bionetworks to track disease progress in Parkinson's patients. Patients download the app, and then are prompted occasionally to complete certain tasks that track how they are doing. These tasks include a walking gait test, a vocal test, and others. Data from the tasks are collected and stored in a database intended for use by researchers, and become part of a broader mission to improve disease tracking in Parkinson's between doctor visits. In the future, if a patient's medicine is not working, the app may signal them that they should see their doctor. 

One of the tests within the mPower app is a memory game, which intends to track how the memory of Parkinson's patients varies on a day-to-day basis. The memory app looks like this:


![mpowerappface.png]({{site.baseurl}}/images/mpowerappface.png)


First, the user is prompted about whether they recently took their Parkinson's medication. Patients can thus be classed into those at their 'best' (just took meds), vs. those at their 'worst' (immediately before taking medication). Then, a 3x3 grid of flowers will appear on the screen, and will light up in a randomized order. After that's done, the user is tasked with touching the flowers in the same order they lit up. If they do poorly, the app reduces the number of flowers to a 2x2 grid. If they do well, then the complexity increases to a 4x4 grid. 

The intent is to change how patients are  and hopefully to be predictive of medical status so that 

in collaboration with Apple, using the Apple HealthKit, a a The focus of my analysis is a game within the mPower iphone app, which tracks 



![confusion_randforest_resampled1.png]({{site.baseurl}}/images/confusion_randforest_resampled1.png)
![park_nopark_resample2_agehist.png]({{site.baseurl}}/images/park_nopark_resample2_agehist.png)
![park_nopark_resample1_agehist.png]({{site.baseurl}}/images/park_nopark_resample1_agehist.png)
![park_nopark_scorehist.png]({{site.baseurl}}/images/park_nopark_scorehist.png)
![park_nopark_agehist.png]({{site.baseurl}}/images/park_nopark_agehist.png)
![sage.png]({{site.baseurl}}/images/sage.png)









