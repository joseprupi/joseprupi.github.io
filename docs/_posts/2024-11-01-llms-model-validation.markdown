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

Ok, we have a base now. We understand that validating models is a regulated process guided by SR 11-7 so let's gather some more detail on what this should be:

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

This seems simple, because it is, and useful, because it also is, but I don't think this is a common practice out of banking. This makes me think that companies in other industries could  benefit from adopting a similar approach. A standardized process that clearly and consistently reports reliability and proper use of the models being used.

And the purpose of the process isn't simply to generate reporting, it's to create better models, and my experience is that it works. Development teams create higher quality products if there exists a process auditing their work and expecting things like justifying their decisions or document in a way that lines of business can understand.

# A model

Now let's try to reproduce how a validation of a model would look like and how LLMs could keep helping us with it. 

First, we need a model to validate. While almost any model could work for demonstration purposes, I wanted to choose something that resembles a real financial case. I searched on Kaggle and found this challenge:

> [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability). 
> 
> **Overview:** The goal of this competition is to predict which clients are more likely to default on their loans. The evaluation will favor solutions that are stable over time.
>
I am not sure how Kaggle works and it is sometimes confusing to me, but I found what appears to be an example notebook that implements a model for the problem, [Home Credit 2024 Starter Notebook](https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook). After downloading the data I have executed the code and it works. 

Now, the first thing I would like to do is understand how the code works, so let's see how LLMs can help us with that. As I mentioned I am not a big LLMs user but with some small research I found that Claude 3.5 Sonnet is the best model to deal with code, or at least is better than Chat GPT, and indeed with some tests I have done it is significantly better.

So let's see what the model has to say about the code. I have converted the notebook into a script (jupyter nbconvert --to script loans.ipynb) copy and paste the result into Claude AI prompt and ask:

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

This is already surprisingly good to me. It explains the code and also gets into some of the model parametrization and how that affects its behavior. I will stop here with the code analysis, but we could definitely keep asking about it. It is even offering more detail. 

Let's now ask for help to the LLM for some tests:

> Given this code to train a credit model, could you create and give me the tests that should be performed according to SR 11-7? The tests should have visualization when possible, the scope should be as broad as possible and the results should be put inside a tex file, including the images. Could you make it in a way that all outputs (tex and images) are in the same folder?

The result is some code that I execute in a cell right after the last one from the Notebook I downloaded. It works straight away, and after compiling the tex file, this is the result:

![image](/assets/output1.png)
![image](/assets/output2.png)
![image](/assets/output3.png)
![image](/assets/output4.png)
![image](/assets/output5.png)

