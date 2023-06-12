---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: Midterm Report
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


### **Data Collection**
The dataset we are using is found at [https://eightportions.com/datasets/Recipes/](https://eightportions.com/datasets/Recipes/) (scroll down to 'Download recipes'). It includes a scraped list of ~125,000 recipes from various websites. We chose this dataset for various reasons:
- The recipes come from a variety of websites, includes recipes from various cuisines, and includes a large number of recipes.
- The recipes include both ingredients and instructions.
- The recipes are in a JSON format that is very simple: string title, string instructions, and a string list of ingredients.

In order to create our cuisine classification algorithm, we also needed a dataset which contained labels for the cuisine of the recipe. Thus, we decided to use [this dataset](https://raw.githubusercontent.com/warcraft12321/HyperFoods/master/data/kaggle_and_nature/kaggle_and_nature.csv) found from Kaggle and Nature, which contains ~100,000 recipes with strictly ingredients and cuisine listed. This would be a sufficient dataset for us to train and test our classification algorithm, and in practice it can also be used on the above dataset to classify the recipes generated.

We also found some alternate datasets we may use in case we run into any problems with the above datasets:
- [https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- [http://pic2recipe.csail.mit.edu/](http://pic2recipe.csail.mit.edu/)
- [https://clickhouse.com/docs/en/getting-started/example-datasets/recipes/](https://clickhouse.com/docs/en/getting-started/example-datasets/recipes/)

#### **Data Cleansing & Pre-processing**
For the Naive Bayes algorithm for labeling a recipe, there were a few steps we needed to do:
- Randomly break the labeled data into two parts: training (70%) and testing (30%)
This was easily accomplished using the following code: 
    - `train_data_raw, test_data_raw = train_test_split(lines_unbiased, test_size=0.3, random_state=42)` 
    - A seed of 42 to was used to ensure repeatable results.
- Avoid biases in training by making sure there were an equal number of labeled datapoints per type of cuisine (600 each). Before doing this step, the training data was extermely biased due to a disproportionate amount of data per type of cuisine. Thus, the model was very inaccurate:
![biased](./images/biased_classifier.png)
    - This was our first attempt and is not the final model.
    - As you can see, 0 cuisines were predicted as MiddleEastern and only 2 were predicated as NorthernEuropean. Meanwhile, a large number of cuisines were falsely predicted as NorthAmerican due to the large bias of NorthAmerican data in the training set.
- After removing biases and ensuring there were an equal amount of data per type of cuisine, the model definitely improved. See the `results and discussion` section for more details.

For the generative model (which is yet to be implemented), there were 4 steps we needed to take in order to clean and pre-process the data for the model:
- Filter out large recipes
- Remove incomplete recipes
- Convert recipe objects into strings

In order to make sure our neural network worked efficiently, we needed to have a predetermined optimal recipe length so that most recipes were included, but we did not have recipes which were too long and would negatively affect the model training. Although our dataset was very clean, we did need to remove some recipes which did not include a name, ingredients, or instructions. We needed to convert recipe objects into strings because the neural network we are using does not understand recipe objects. We picked a simple, large dataset which was very easy to understand, so we did not need to anything other than these 3 steps to ensure that our model would be successful. 

#### **Visualizations**
For our visualizations, we chose to filter out the ten most popular ingredients for each cuisine and plot a bar graph of their occurences in the raw dataset.
We implemented these graphs to visualize how a particular ingredient's popularity could skew the results of the classification algorithm between different cuisines. We created these bar graphs for each cuisine. Below are two bar graphs that we plotted for two different cuisines: East Asian and Western European:

![biased](./images/EastAsian_Ingredients.png) ![biased](./images/WesternEuropean_Ingredients.png)

For the training data, which has 600 recipies of each cuisine to avoid bias but also have a large enough dataset, we created the same graphs again in order to check if the results differed. The popularity for ingredients changed, as seen below. This represents how biased data can skew the occurences for each ingredient, and therefore the final results of our classification model.

![biased](./images/EastAsian_Ingredients_Unbiased.png) ![biased](./images/WesternEuropean_Ingredients_Unbiased.png)

As different cuisines have different ingredients, we also wanted to see if the variability of ingredients could affect the classification of recipes. From the following figure, we can see that Middle Eastern and North American recipes have the least variability in the ingredients required to make a dish, whereas Southeast Asian and East Asian cuisines have similar variability, as do European cuisines.

![biased](./images/NumIngredients.png)

### **Results and Discussion**
As mentioned in the previous section, we had to reduce the labeled data set down to 600 pieces of data per type of cuisine to avoid biases in the cuisine classification algorithm. Here is the outcome of the model after accounting for biases:
![unbiased](./images/unbiased_classifier.png)
- As you can see, there is a diagonal line on the heatmap, proving that this model is much better than the previous iteration.
- We can see that the model has a 64.39% accuracy while the previous iteration had a 75.13% accuracy. However, this the accuracy of the old model is misleading because it was also tested on biased data. 
- When testing the old model with unbiased testing data, the actual accuracy drops down to 59.2% instead of 75.13%. Thus, the new model is better.
- Still, a 64% accuracy is not great. The model could be improved with better testing data or by using an alternate classification method such as Neural Networks. We will look into improving this before the final report.

The recipe generation part of this project will be completed in the Final Report. For the midterm report, we primarily focused on cuisine classification and data cleansing. 


### **References**
[1] Lichtenstein, Alice. “28% Of Americans Cant Cook.” Tufts Health &amp; Nutrition Letter, 17 Sept. 2019, https://www.nutritionletter.tufts.edu/general-nutrition/28-of-americans-cant-cook/. 

[2] Martin, Emmie. “90% Of Americans Don't like to Cook - and It's Costing Them Thousands Each Year.” USA Today, Gannett Satellite Information Network, 27 Sept. 2017, https://www.usatoday.com/story/money/personalfinance/2017/09/27/90-americans-dont-like-cook-and-its-costing-them-thousands-each-year/708033001/. 

[3] Ma, Peihua, et al. “Neural Network in Food Analytics.” Critical Reviews in Food Science and Nutrition, 2022, pp. 1–19., https://doi.org/10.1080/10408398.2022.2139217. 

### **Proposed Timeline**
[https://gtvault-my.sharepoint.com/:x:/g/personal/psheth32_gatech_edu/EcPfQOCUR1FCm8s1EGD6SIQBviLuYOnGql_zA2mVGB5yCw](https://gtvault-my.sharepoint.com/:x:/g/personal/psheth32_gatech_edu/EcPfQOCUR1FCm8s1EGD6SIQBviLuYOnGql_zA2mVGB5yCw)


### **Contribution Table**

| Name | Contributions |
| --- | --- |
| Kevin | Data Cleaning, Data Preprocessing |
| Nick | Classification Algorithm, Results |
| Erland | Data Cleaning, Data Preprocessing  |
| Pallavi | Visualizations, Contribution Table |
| Prisha | Visualizations, Contribution Table |
