---
layout: post
title: "Using LLMs for Model Validation"
date: 2024-11-01 00:00:00 -0000
categories: misc
---

# Intro

Although it might seem straightforward, let's begin with what I mean by model validation in this post. Yes, it involves ensuring that models function correctly, a broad topic in itself, but this is specifically focused within the US banking environment.

So, what does Chat GPT say about model validation in this context?

> **Me:** What does model validation mean in the US banking environment? Could you summarize it with few sentences?
>  
> **ChatGPT:** In the U.S. banking environment, model validation refers to the rigorous process of evaluating financial models to ensure they are reliable, accurate, and compliant with regulatory standards. It involves assessing the model's performance, assumptions, limitations, and potential risks. This practice is essential for managing model risk and adhering to regulatory guidelines such as SR 11-7, which requires banks to maintain robust model risk management frameworks to prevent financial and operational issues.

Alright, we have a base now. We understand that validating models is a regulated process guided by SR 11-7 so let's gather some more detail on what this should be:

> **Me:** What should a model validation contain according to SR 11-7?
>  
> **ChatGPT:** According to SR 11-7, a comprehensive model validation should include the following key components:
>
>* **Evaluation of Conceptual Soundness:**
 An in-depth assessment of the model's design, theory, and logic, including underlying assumptions and methodology. This ensures that the model is appropriate for its intended purpose.
>
>* **Ongoing Monitoring:** Regular checks to confirm that the model's performance aligns with expectations over time. This includes tracking key performance metrics and reviewing outcomes to identify any significant deviations.
>
>* **Outcomes Analysis:** A quantitative comparison of the model's predictions against actual outcomes. This step helps assess the accuracy and reliability of the model in real-world conditions.
>
>* **Benchmarking and Back-Testing:** Comparing the model's results with alternative models and validating performance through back-testing with historical data.
>
>* **Assessment of Limitations and Uncertainty:** Identifying potential limitations in the model's framework and quantifying any associated uncertainties or risks.
>
>These elements ensure a robust validation process, enabling institutions to identify and mitigate model risks effectively, maintaining compliance with regulatory requirements.

And what form and structure does a model validation take?

> **Me:** How should these results be presented?
>  
>Model validation results, in line with SR 11-7, should be >presented in a structured, transparent, and comprehensive >report, typically covering the following sections:
>
>* **Executive Summary**: A concise overview of the validation's key findings, model strengths and weaknesses, and an overall assessment of the model’s risk level and validation outcome (e.g., approved, approved with conditions, or not approved).
>
>* **Scope and Objectives**: A clear outline of the validation’s purpose, including the model’s intended use, its assumptions, and limitations. It should specify the validation scope, such as whether it's a full-scope or targeted validation.
>
>* **Methodology and Approach**: A description of the validation process, including the methods, tools, and benchmarks used. This part should clarify how the validation aligns with regulatory requirements and internal policies.
>
>* **Evaluation of Conceptual Soundness**: Findings on the model’s design, methodology, assumptions, and overall soundness. This section often includes a detailed review of data sources, model components, and theoretical foundations.
>
>* **Ongoing Monitoring and Performance Review**: Insights into the model’s stability, accuracy, and predictive power over time, along with performance metrics and any monitoring results that highlight trends or areas of concern.
>
>* **Outcomes Analysis and Back-Testing Results**: Results of outcome analysis, including back-testing findings that compare predicted versus actual results. This section should provide quantitative and qualitative analyses, highlighting any discrepancies and their implications.
>
>* **Issues and Recommendations**: Identification of any issues, weaknesses, or limitations, along with prioritized recommendations for improvements. This can include both short-term remediation actions and longer-term improvements to address identified risks.
>
>* **Conclusion and Sign-Off**: A summary of key points and the final opinion on the model’s viability, along with sign-offs from validation team members and relevant stakeholders.
>
>* **Supporting Documentation**: Additional materials such as data sources, code, validation testing details, and other relevant documents to ensure transparency and reproducibility.
>
>Effective presentation should ensure clarity for stakeholders, including management and regulators, with visual aids like charts and tables where helpful.

Ok, so validating models in banking is assessing they work properly and then write the results in a report. 

While this process seems straightforward, it kind of is, it’s also incredibly useful. And it might be a topic for a different writing, but I don’t believe this practice is common outside of banking. And this makes me think that other industries could benefit from adopting a similar approach: a standardized process that ensures models are reliable and used appropriately, with clear and consistent reporting.

Importantly, the goal of this process isn’t just to produce reports, it’s to improve the quality of models. And In my experience works. Development teams tend to build better products when they know their work will be audited, requiring them to justify decisions and document their processes in a way that business stakeholders can easily understand.

# A model

Let's now try to reproduce how a validation of a model would look like and how LLMs could assist with it. 

First, we need a model to validate. While almost any model could work for demonstration purposes, I wanted to choose something that resembles a real financial case. I searched on Kaggle and found this challenge:

> [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability). 
> 
> **Overview:** The goal of this competition is to predict which clients are more likely to default on their loans. The evaluation will favor solutions that are stable over time.
>
I am not sure how Kaggle works and it is sometimes confusing to me, but I found what appears to be an example notebook that implements a model for the problem, [Home Credit 2024 Starter Notebook](https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook). After downloading the data I have executed the code and it works. 

Having a high level understanding of the implementation it is one of the first and fundamental steps in a model validation. LLMs might be able to help with this, and I will use Claude 3.5 Sonnet as seems to perform well with code related tasks, or at least is better than Chat GPT, which I found to be true after some tests where it proved to be significantly better.

Let’s see what the model can tell us about the code. I’ve converted the Jupyter notebook into a script using the command *jupyter nbconvert --to script loans.ipynb*. Then, I copied and pasted the resulting script into Claude AI’s prompt and asked:

> What is this code doing?
> Let me break down this code - it appears to be implementing a credit risk or default prediction model with a focus on stability across time. Here are the key components:
>
> 1 Data Loading and Preprocessing:
> * Loads several CSV files containing different types of data (base tables, static data, credit bureau data, person data)
> * Uses Polars (a fast DataFrame library) for initial data processing
> * The data appears to be related to loan applications, with different tables containing various attributes about applicants
>
> 2 Feature Engineering:
> * Creates aggregated features from person-related data:
> &nbsp;&nbsp;&nbsp;&nbsp;- Maximum occupation income
&nbsp;&nbsp;&nbsp;&nbsp;- Whether any applicant is self-employed
&nbsp;&nbsp;&nbsp;&nbsp;- Housing type for primary applicant
> * Creates credit bureau features:
&nbsp;&nbsp;&nbsp;&nbsp;- Maximum payments overdue
&nbsp;&nbsp;&nbsp;&nbsp;- Whether payments were overdue by more than 31 days
>
> 3 Model Training:
> * Uses LightGBM, a gradient boosting framework
> * Splits data into train (60%), validation (20%), and test (20%) sets
> * Trains a binary classification model with AUC (Area Under Curve) as the metric
> * Uses early stopping to prevent overfitting
>
> 4 Model Evaluation:
> 
> * Calculates AUC scores for train, validation, and test sets
> * Implements a custom stability metric called gini_stability that:
&nbsp;&nbsp;&nbsp;&nbsp;- Tracks model performance (Gini coefficient) across different time periods (WEEK_NUM)
&nbsp;&nbsp;&nbsp;&nbsp;- Penalizes: 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Declining performance over time (negative slope in performance)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- High variance in performance (standard deviation of residuals)
&nbsp;&nbsp;&nbsp;&nbsp;- This suggests the model is being used in a production setting where stable predictions over time are important
>
> 5 The hyperparameters used suggest a relatively conservative model:
>* Max depth of 3
>* Learning rate of 0.05
>* Feature and bagging fractions < 1 for regularization
>
>Production ML Pipeline with Stability MetricsClick to open code
This code appears to be part of a production system where model stability over time is as important as raw predictive performance. The stability metric is particularly interesting as it penalizes both declining performance over time and high variance in performance, which are important considerations for deployed models.
>
>Would you like me to explain any particular aspect in more detail? For example, I could dive deeper into:
>
>1 The stability metric calculation
2 The feature engineering approach
3 The model hyperparameter choices

This is already surprisingly good to me. It explains the code and also gets into some of the model parametrization and how that affects its behavior. I will stop here with the code analysis, but I feel we could definitely keep asking about it to get more details.

Let's now ask for help to the LLM for some tests:

> Given this code to train a credit model, could you create and give me the tests that should be performed according to SR 11-7? The tests should have visualization when possible, the scope should be as broad as possible and the results should be put inside a tex file, including the images. Could you make it in a way that all outputs (tex and images) are in the same folder?

The result from this request is some code that I execute in a cell right after the last one from the Notebook I downloaded. It works straight away, and after compiling the tex file, this is the result:

![image](/assets/output1.jpg)
![image](/assets/output2.jpg)
![image](/assets/output3.jpg)
![image](/assets/output4.jpg)
![image](/assets/output5.jpg)

If the code and results are accurate, it's impressive. In just five minutes, I have a report that provides insights into fundamental aspects of the model.

While it is true that the report lacks detailed information, such as what the feature importance scores are, it also offers valuable data like feature correlations, model performance across datasets, and temporal performance trends

Now, I'm sure some might find these tests simple and barely useful, but just taking a look at some of the submitted Kaggle solutions and my experience, I can tell that the needs are completely different to the credit models in finance I have been working with. 

For example, here it seems a matter of getting the maximum score in the competition, meaning that having an ensemble of complicated models with as many features as possible is the best way to go, my experience with credit models is that simplicity is highly valuable: there are compliance requirements and the users of the model that want to understand what the it is doing.

And the reason I am saying this is because the plot of the correlation matrix is straight away saying that the dark islands would require further assessment. This is a valid and actually a real test I would conduct to a model like this.

It is also interesting to see the PSI over time plot along with the average score trend. It seems that starting at week 60 something something is changing in the behavior of the model, and this is also something important to understand with credit models. It would depend on a lot of factors and this is a simplified example, but understanding what is going on with the model and what made its behavior change is important. 

In summary, these tests are legit. It took just five minutes to "code", execute, and compile them into a PDF document, which is cool.

* Features correlation, which would require more testing from a model validation analyst perspective. Simplicity is key in models so I would definitely require a justification for having all of them in the model. Adding features is usually a problem in finance as makes the model more complicated which is something specially worst in credit as users usually want to understand what the model is doing, and there are also some compliance issues involved making credit models too complicated.
* Basic performance metrics for training and testing data.
* Performance of the model in time. I find this to be specially interesting 

I also find the PSI over time a good test.