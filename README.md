# Diamond-Price-Prediction-Model
Diamonds Price Prediction model - Scikit ML Library

Diamonds in Depth analysis and Prediction Model
About the Research Question and Objectives
About the Research Question
Diamonds are the one most valuable stones found in earth. This question need in depth idea about Diamonds. As an example we have to learn about the each characteristics separately and with coupling.
About the Objectives
Our objective was to achieve an in depth analysis about Diamonds by comparing each characteristics, how they are correlated with each other. Other main objective was making a predictive model for diamonds. When we enter characteristics using the model we can get a predictive price for the Diamond.
Statistical Techniques Used
•	Correlational analysis 
•	Linear Regression
•	Lasso Regression
•	Ridge Regression  
•	RandomForest Regression 
•	KNeighbours Regression
Findings to support your objective
Correlation Between the characteristics
Correlation is one of the best way and the initial step to start the analysis. With this method can get an overall idea of the characteristics.
These are some conclusions made by this method
1.	Depth is inversely related to Price.
•	This is because if a Diamond's Depth percentage is too large or small the Diamond will become 'Dark' in appearance because it will no longer return an Attractive amount of light.
2.	The Price of the Diamond is highly correlated to Carat, and its Dimensions.
3.	The Weight (Carat) of a diamond has the most significant impact on its Price.
•	Since, the larger a stone is, the Rarer it is, one 2 carat diamond will be more 'Expensive' than the total cost of two 1 Carat Diamonds of the same Quality.
4.	The Length(x), Width(y) and Height (z) seems to be highly related to Price and even each other.
5.	Self-Relation ex. Of a feature to itself is 1 as expected.
Carat vs Price
•	Carat refers to the Weight of the Stone, not the Size.
•	The Weight of a Diamond has the most significant Impact on its Price.
•	Since the larger a Stone is, the Rarer it is, one 2 Carat Diamond will be more Expensive than the Total cost of two 1 Carat Diamonds of the Same Quality.
•	Carat varies with Price Exponentially.
Cut vs Price
•	The Cut can still Drastically Increase or Decrease its value. With a Higher Cut Quality, the Diamond’s Cost per Carat Increases.
•	Premium Cut on Diamonds as we can see are the most Expensive, followed by Excellent / Very Good Cut.
Color vs Price
•	The Color of a Diamond refers to the Tone and Saturation of Color, or the Depth of Color in a Diamond.
•	The Color of a Diamond can Range from Colorless to a Yellow or a Faint Brownish Colored hue.
•	Colorless Diamonds are rarer and more valuable because they appear Whiter and Brighter.
Clarity vs Price
•	Diamond Clarity refers to the absence of the Inclusions and Blemishes.
•	It seems that VS1 and VS2 affect the Diamond's Price equally having quite high Price margin.
Depth vs Price
•	The Depth of a Diamond is its Height (in mm) measured from the Culet to the Table.
•	We can infer from the plot that the Price can vary heavily for the same Depth.
•	The Pearson's Correlation shows that there's a slightly inverse relation between the two.

Table vs Price
•	Table is the Width of the Diamond's Table expressed as a Percentage of its Average Diameter.
•	If the Table (Upper Flat Facet) is too large then light will not play off of any of the Crown's angles or facets and will not create the Sparkly Rainbow Colors.
•	If it is too small then the light will get Trapped and that Attention grabbing shaft of light will never come out but will “leak” from other places in the Diamond.
Volume vs Price
•	As the Dimensions increases, obviously the Prices Rises as more and more Natural Resources are Utilized.
•	It seems that there is Linear Relationship between Price and Volume (x * y * z).
Find the suitable Regression Algorithm
To find the best regression algorithm we have entered several Regression Algorithms to find the new one. First create Training and Testing dataset from the dataset that we got.
We test for the mean squared error. So we can choose the lowest mean squared error algorithm to make our prediction model  
Create the prediction model using the best regression model. We received the Random Forest Regressor as the best algorithm to use.
  
Limitations of the study

•	Scikit learn kit have drawbacks when importing models ex: Robust Scaler, Standard Scaler
•	Mainly we can’t use the data as it is. For that we have to create a model that we can use for the prediction.
Example : ['cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2']
•	Dataset is small and limited. Model accuracy is based on the dataset. To make more accurate predictions have to have much bigger dataset than this.
•	Only tested for several Regression algorithms so accuracy is calculated according to those tested algorithms.

