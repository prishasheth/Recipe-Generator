---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: Proposal
---
### **Team Members**
- Kevin Hoxha
- Erland Mulaokar
- Nicholas Hulston
- Pallavi Eranezhath
- Prisha Sheth

### **Introduction/Background**
Many Americans today do not have a lot of experience cooking. A survey conducted by Impulse Research found that about 28% of Americans say they do not know how to cook, and purchase the majority of their meals [1]. This can be a very expensive habit: the average household in America spends $3,008 per year on dining out [2]. Our group is interested in creating a tool that allows users to generate recipes for food they would like to cook based on ingredients that the user has available to them and the user’s taste preferences, with the ultimate goal of making cooking easier for the average person.


### **Problem Definition**
It is hard for many young people to get started in the kitchen. Although most Americans would like to cook, many are deterred by the challenge of finding food that is easy to make and tastes good. We want to train a machine learning algorithm that will generate recipes to users based on their favorite foods and the ingredients they have available to them. This way, personally tailored recipes are only a search away.


### **Methods**
The primary method we will be using is LSTM (Long Short-Term Memory) Neural Networks. Using an LTSM Neural Network would be a good choice because:
- LSTMs are a popular choice for models that generate text [3].
- LSTMs are great at generating new sequences by sampling from the training data.
- LSTMs are able to handle flexible input. Some users may have few inputs, while other users might have a large amount of input due to dietary restrictions and preferences.
- Neural networks are great at handling large amounts of data.

Another important feature of our project will be the ability to choose specific cuisines the user is interested, such as Italian, Chinese, Mediterranean, etc. While generated recipes will not truly belong to a specific cuisine, we can use a classification algorithm to determine the best-fit for each recipe. We can use Naive Bayes since it is fast, simple, and a popular classification algorithm for text inputs.


### **Datasets**
The dataset we plan on using is found at [https://eightportions.com/datasets/Recipes/](https://eightportions.com/datasets/Recipes/) (scroll down to 'Download recipes'). It includes a scraped list of ~125,000 recipes from various websites. We chose this dataset for various reasons:
- The recipes come from a variety of websites, includes recipes from various cuisines, and includes a large number of recipes.
- The recipes include both ingredients and instructions.
- The recipes are in a JSON format that is very simple: string title, string instructions, and a string list of ingredients.

However, we also found some alternate datasets we may use in case we run into any problems with the above dataset:
- [https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- [http://pic2recipe.csail.mit.edu/](http://pic2recipe.csail.mit.edu/)
- [https://clickhouse.com/docs/en/getting-started/example-datasets/recipes/](https://clickhouse.com/docs/en/getting-started/example-datasets/recipes/)

### **Potential Results and Discussion**

We hope to build a robust recommendation system based on the above methods in order to generate recipes catering to an individual’s personal preferences and available ingredients. However to determine the validity of our results, we will use metrics such as precision and recall, F1-score, silhouette coefficient, elbow method, and the Davies–Bouldin index (DBI).


### **References**
[1] Lichtenstein, Alice. “28% Of Americans Cant Cook.” Tufts Health &amp; Nutrition Letter, 17 Sept. 2019, https://www.nutritionletter.tufts.edu/general-nutrition/28-of-americans-cant-cook/. 

[2] Martin, Emmie. “90% Of Americans Don't like to Cook - and It's Costing Them Thousands Each Year.” USA Today, Gannett Satellite Information Network, 27 Sept. 2017, https://www.usatoday.com/story/money/personalfinance/2017/09/27/90-americans-dont-like-cook-and-its-costing-them-thousands-each-year/708033001/. 

[3] Ma, Peihua, et al. “Neural Network in Food Analytics.” Critical Reviews in Food Science and Nutrition, 2022, pp. 1–19., https://doi.org/10.1080/10408398.2022.2139217. 

### **Proposed Timeline**
[https://gtvault-my.sharepoint.com/:x:/g/personal/psheth32_gatech_edu/EcPfQOCUR1FCm8s1EGD6SIQBviLuYOnGql_zA2mVGB5yCw](https://gtvault-my.sharepoint.com/:x:/g/personal/psheth32_gatech_edu/EcPfQOCUR1FCm8s1EGD6SIQBviLuYOnGql_zA2mVGB5yCw)


### **Contribution Table**

| Name | Contributions |
| --- | --- |
| Kevin | Introduction, Problem Definition, References, Video, GitHub Creation |
| Nick | Methods, Checkpoint, References, Video |
| Erland | Methods, Checkpoint, References, Video |
| Pallavi | Potential Results, Contribution Table, Timeline, References, Video |
| Prisha | Potential Results, Contribution Table, Timeline, References, Video |

### **Presentation Video**
[![CS 4641 Group 8: Project Proposal](http://img.youtube.com/vi/ba_lQ9t0Blo/0.jpg)](http://www.youtube.com/watch?v=ba_lQ9t0Blo "CS 4641 Group 8: Project Proposal")

