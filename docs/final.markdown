---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: Final Report
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
The primary method we will be using is LSTM (Long Short-Term Memory) Neural Networks. Using an LSTM Neural Network would be a good choice because:
- LSTMs are a popular choice for models that generate text [3].
- LSTMs are great at generating new sequences by sampling from the training data.
- LSTMs are able to handle flexible input. Some users may have few inputs, while other users might have a large amount of input due to dietary restrictions and preferences.
- Neural networks are great at handling large amounts of data.

Another important feature of our project will be the ability to choose specific cuisines the user is interested, such as Italian, Chinese, Mediterranean, etc. While generated recipes will not truly belong to a specific cuisine, we can use a classification algorithm to determine the best-fit for each recipe. We initially chose to use Naive Bayes (for the midterm report) since it is fast, simple, and a popular classification algorithm for text inputs. However, we found the Naive Bayes to have low accuracy, so we decided to use a Neural Network (for the final report) in hopes of increasing accuracy.


### **Data Collection**
The dataset we are using is found at [https://eightportions.com/datasets/Recipes/](https://eightportions.com/datasets/Recipes/) (scroll down to 'Download recipes'). It includes a scraped list of ~125,000 recipes from various websites. We chose this dataset for various reasons:
- The recipes come from a variety of websites, includes recipes from various cuisines, and includes a large number of recipes.
- The recipes include both ingredients and instructions.
- The recipes are in a JSON format that is very simple: string title, string instructions, and a string list of ingredients.

In order to create our cuisine classification algorithm, we also needed a dataset which contained labels for the cuisine of the recipe. Thus, we decided to use [this dataset](https://raw.githubusercontent.com/warcraft12321/HyperFoods/master/data/kaggle_and_nature/kaggle_and_nature.csv) found from Kaggle and Nature, which contains ~100,000 recipes with strictly ingredients and cuisine listed. This would be a sufficient dataset for us to train and test our classification algorithm, and in practice it can also be used on the above dataset to classify the recipes generated.

#### **Data Cleansing & Pre-processing**
For the Naive Bayes and Neural Net algorithms for labeling a recipe, there were a few steps we needed to do:
- Randomly break the labeled data into two parts: training (70%) and testing (30%)
This was easily accomplished using the following code: 
    - `train_data_raw, test_data_raw = train_test_split(lines_unbiased, test_size=0.3, random_state=42)` 
    - A seed of 42 to was used to ensure repeatable results.
- Avoid biases in training by making sure there were an equal number of labeled datapoints per type of cuisine (600 each). Before doing this step, the training data was extremely biased due to a disproportionate amount of data per type of cuisine. Thus, the model was very inaccurate:
![biased](./images/biased_classifier.png)
    - This was our first attempt and is not the final model.
    - As you can see, 0 cuisines were predicted as MiddleEastern and only 2 were predicated as NorthernEuropean. Meanwhile, a large number of cuisines were falsely predicted as NorthAmerican due to the large bias of NorthAmerican data in the training set.
- After removing biases and ensuring there were an equal amount of data per type of cuisine, the model definitely improved. See the "Results and Discussion" section for more details.

For the Neural Net classifier, there were a few steps to preprocess the data.
1. Balance the data, similar to what was done for the Naive Bayes classifier
2. Tokenize the data to keep only frequent vocabulary and standardize the data
3. Encode the labels to simplify loss calculation

        # Balance the data
        class_counts = Counter(labels)
        majority_count = max(class_counts.values())
        minority_ratio = 0.5
        resampled_descriptions = []
        resampled_labels = []
        for class_label, count in class_counts.items():
            target_count = int(majority_count * minority_ratio) if count < majority_count else count
            resampled_class_descriptions, resampled_class_labels = resample(
                descriptions[labels == class_label],
                labels[labels == class_label],
                replace=True,
                n_samples=target_count,
            )
            resampled_descriptions.extend(resampled_class_descriptions)
            resampled_labels.extend(resampled_class_labels)
        descriptions = np.array(resampled_descriptions)
        labels = np.array(resampled_labels)

        # Tokenize
        tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
        tokenizer.fit_on_texts(descriptions)
        sequences = tokenizer.texts_to_sequences(descriptions)
        padded_sequences = pad_sequences(sequences, padding='post')

        # Encode labels
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)

For the generative model, there were 3 steps we needed to take in order to clean and pre-process the data for the model:
- Filter out large recipes
- Remove incomplete recipes
- Convert recipe objects into strings

In order to make sure our LSTM neural network worked efficiently, we needed to have a predetermined optimal recipe length so that most recipes were included, but we did not have recipes which were too long and would negatively affect the model training. To decide on this length, we plotted a distribution of recipe lengths in our dataset and found that a character limit of $$\sim$$ 2000 characters would cover about 100,000 of the 125,000 recipes in the dataset. Although our dataset was very clean, we did need to remove some recipes which did not include a name, ingredients, or instructions. We needed to convert recipe objects into strings because the neural network we are using does not understand recipe objects. We picked a simple, large dataset which was very easy to understand, so we did not need to anything other than these 3 steps to ensure that our model would be successful. Each recipe was also padded with "$$\sim$$", the character we chose to use as a signal to the LSTM model that the recipe was finished. Recipes were padded to each have exactly 2000 characters. After converting each recipe to a string object, each recipe in our dataset matched the following format:

        TITLE: Slow Cooker Chicken and Dumplings

        INGREDIENTS:
        - 4 skinless, boneless chicken breast halves 
        - 2 tablespoons butter 
        - 2 (10.75 ounce) cans condensed cream of chicken soup 
        - 1 onion, finely diced 
        - 2 (10 ounce) packages refrigerated biscuit dough, torn into pieces 

        INSTRUCTIONS:
        - Place the chicken, butter, soup, and onion in a slow cooker, and fill with enough water to cover.
        - Cover, and cook for 5 to 6 hours on High. About 30 minutes before serving, place the torn biscuit dough in the slow cooker. Cook until the dough is no longer raw in the center.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After converting each recipe to a string, in order for the LSTM model to work correctly, we would have to vectorize each recipe. This is because the LSTM predicts on a character-level, so each character in our alphabet must be assigned a corresponding integer. This was done as follows:

    STOP_CHAR = '~'
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        char_level=True,
        filters='',
        lower=False,
        split=''
    )
    tokenizer.fit_on_texts([STOP_CHAR])
    tokenizer.fit_on_texts(dataset_filtered)
    dataset_vectorized = tokenizer.texts_to_sequences(dataset_filtered)

Then each recipe in `dataset_vectorized` was converted to something in the form `[29, 30, 29, 46, 37, 35, 1, 31, 10, 5, ...]`. 

#### **Visualizations**
For our visualizations, we chose to filter out the ten most popular ingredients for each cuisine and plot a bar graph of their occurences in the raw dataset.
We implemented these graphs to visualize how a particular ingredient's popularity could skew the results of the classification algorithm between different cuisines. We created these bar graphs for each cuisine. Below are two bar graphs that we plotted for two different cuisines: East Asian and Western European:

![biased](./images/EastAsian_Ingredients.png) ![biased](./images/WesternEuropean_Ingredients.png)

For the training data, which has 600 recipies of each cuisine to avoid bias but also have a large enough dataset, we created the same graphs again in order to check if the results differed. The popularity for ingredients changed, as seen below. This represents how biased data can skew the occurences for each ingredient, and therefore the final results of our classification model.

![biased](./images/EastAsian_Ingredients_Unbiased.png) ![biased](./images/WesternEuropean_Ingredients_Unbiased.png)

As different cuisines have different ingredients, we also wanted to see if the variability of ingredients could affect the classification of recipes. From the following figure, we can see that Middle Eastern and North American recipes have the least variability in the ingredients required to make a dish, whereas Southeast Asian and East Asian cuisines have similar variability, as do European cuisines.

![biased](./images/NumIngredients.png)

### **Results and Discussion**

#### **Classification Model**
As mentioned in the previous section, we had to reduce the labeled data set down to 600 pieces of data per type of cuisine to avoid biases in the Naive Bayes cuisine classification algorithm. Here is the outcome of the model after accounting for biases:

![unbiased](./images/unbiased_classifier.png)
- As you can see, there is a diagonal line on the heatmap, proving that this model is much better than the previous iteration.
- We can see that the model has a 64.39% accuracy while the previous iteration had a 75.13% accuracy. However, this the accuracy of the old model is misleading because it was also tested on biased data. 
- When testing the old model with unbiased testing data, the actual accuracy drops down to 59.2% instead of 75.13%. Thus, the new model is better.
- Still, a 64% accuracy is not great. The model could be improved with better testing data or by using an alternate classification method such as Neural Networks.

For the final report, we decided to reattempt the classification of recipes using a Neural Network instead and see if the results improved.

After experimenting with the parameters and layers of the model, we settled on 10 epochs. This seemed to be a good balance to avoid overfitting but improve accuracy. We also used the following layers:

        model = tf.keras.Sequential()
        model.add(Embedding(10000, 100, input_length=padded_sequences.shape[1]))
        model.add(GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(len(encoder.classes_), activation='softmax'))

After training and testing the model, we reached an accuracy of 76.46%:
![neural network](./images/neural_net_classifier.png)

This is a significant improvement compared to the Naive Bayes accuracy of 64.39%. It seems a large portion of the drop in accuracy is due to a bias in predicting cuisines as NorthAmerican (notice the subtle but significant horizontal line of color). This could likely be improved by using better and more unbiased training data.

<img src="./images/loss.png" alt="neural network" width="500">

The above graph demonstrates the overall loss over ten epochs. As the number of epochs increases, the overall loss for the training set decreases, which is consistent with the following graph for accuracy of the training set. As the number of epochs increases, the overall accuracy of the training set increases.

<img src="./images/accuracy.png" alt="neural network" width="500">

This is an example of the output of the neural network classifier. Our model chooses 10 recipes at random and predicts the cuisine it would be classified as. 
![generative](./images/recipeTable.png)

#### **Generative Model**
Our generative model was built using TensorFlow Keras with the following layers. The Embedding layers were used for the input in order to vectorize the inputs to our model, while the LSTM layer is the actual neural network, and the Dense layer is the output layer.

![diagram](./images/model.png)

In order to train our LSTM generative model, we had to settle on hyperparameters that would produce the least possible loss. After some trial an error, we eventually decided on the following parameters:
    
    batch_size = 64
    embedding_dim = 256
    rnn_units = 1024
    epochs = 20
    steps_per_epoch = 50

These parameters allowed for the training algorithm to run in a timely manner and avoid overfitting, while also producing a loss value of about 0.75. See the graph below

![loss](./images/generator_loss.png)

After training and fitting our model, we were able to generate new recipes when prompting the model with a keyword. For example, when inputting the word "Banana" to start the recipe, our predictive model generated the following recipe on a character-by-character basis:

    TITLE: Banana Sugar Cookies

    INGREDIENTS:
    - 1 cup sugar
    - 1 cup all-purpose flour
    - 1 teaspoon baking powder
    - 1/2 teaspoon salt
    - 1/2 cup milk
    - 1 large egg
    - 1 cup semisweet chocolate chips

    INSTRUCTIONS:
    - Preheat oven to 350 degrees F.
    - In a large mixing bowl, combine flour, baking powder, salt and sugar. Mix well. Add egg and milk and mix well. Add flour mixture to butter mixture and stir until well combined. Stir in chocolate chips and chocolate chips. Drop by teaspoonfuls onto prepared baking sheet. Bake for 10 to 12 minutes or until golden brown. Cool on wire racks.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
This recipe seems to be feasible and looks like something that would turn out well if actually baked. However, other recipes generated did not turn out as well, as seen below when "Breakfast" was input into the model.

    TITLE: Breakfast feracher with Chocolate Chips

    INGREDIENTS:
    - 1 cup sugar
    - 1 cup water
    - 1 tablespoon vanilla extract
    - 1/2 cup chopped pecans
    - 1 cup chopped pecans
    - 1 cup chopped pecans
    - 1 cup almonds, toasted
    - 1 cup chopped pecans
    - 1 cup chopped pecans
    - 1 cup chopped pecans
    - 1 cup chopped pecans
    - 1 cup sugar
    - 1 cup chopped pecans
    - 1 cup chopped pecans

    INSTRUCTIONS:
    - Preheat the oven to 350 degrees F.
    - In a large bowl, mix together the flour, sugar, and salt. Add the butter and mix until combined. Add the butter and mix well. Add the butter and mix until the dough comes together. Divide the dough into 4 equal parts. Place the dough on a floured surface and place it on a floured surface. Cut the dough into 1/2-inch rounds. Place the cookies on a baking sheet and bake until the cookies are golden brown, about 12 to 14 minutes. Remove the cookies from the oven and let cool on a wire rack.
    - For the filling: In a large bowl, whisk together the egg yolks, sugar, and vanilla until smooth. Stir in the

It was somewhat common for ingredients to be repeated, recipe instructions to not make much sense, or for recipes to omit either the ingredients or instructions sections. In order to improve our model in the future, we can train the model with more epochs and more steps per epoch in order to further reduce loss. However, this may lead to more overfitting and no true generation of recipes but rather repeating existing recipes.

### **References**
[1] Lichtenstein, Alice. “28% Of Americans Cant Cook.” Tufts Health &amp; Nutrition Letter, 17 Sept. 2019, https://www.nutritionletter.tufts.edu/general-nutrition/28-of-americans-cant-cook/. 

[2] Martin, Emmie. “90% Of Americans Don't like to Cook - and It's Costing Them Thousands Each Year.” USA Today, Gannett Satellite Information Network, 27 Sept. 2017, https://www.usatoday.com/story/money/personalfinance/2017/09/27/90-americans-dont-like-cook-and-its-costing-them-thousands-each-year/708033001/. 

[3] Ma, Peihua, et al. “Neural Network in Food Analytics.” Critical Reviews in Food Science and Nutrition, 2022, pp. 1–19., https://doi.org/10.1080/10408398.2022.2139217. 

### **Proposed Timeline**
[https://gtvault-my.sharepoint.com/:x:/g/personal/psheth32_gatech_edu/EcPfQOCUR1FCm8s1EGD6SIQBviLuYOnGql_zA2mVGB5yCw](https://gtvault-my.sharepoint.com/:x:/g/personal/psheth32_gatech_edu/EcPfQOCUR1FCm8s1EGD6SIQBviLuYOnGql_zA2mVGB5yCw)


### **Contribution Table**

| Name | Contributions |
| --- | --- |
| Kevin | Generative LSTM Model |
| Nick | Naive Bayes and Neural Net Classification Algorithms |
| Erland | Generative LSTM Model  |
| Pallavi | Visualizations, Contribution Table |
| Prisha | Visualizations, Contribution Table |

### **Presentation Video**

[![CS 4641 Group 8: Final ](http://img.youtube.com/vi/vx9jovW9qYw/0.jpg)](https://www.youtube.com/watch?v=vx9jovW9qYw "CS 4641 Group 8: Final Project Video")
