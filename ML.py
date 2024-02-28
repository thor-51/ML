#A Machine Learning program to cluster texts pertaining to food poisoning, and providing an accurate brief on it with a graph

from sklearn.cluster import KMeans #For KMeans clustering (to cluster the text data)
from sklearn.feature_extraction.text import TfidfVectorizer #Converting a collection of raw documents into a matrix of TF-IDF features 
import matplotlib.pyplot as plt #Plotting a scatter plot of the data points colored by their cluster assignments
import seaborn as sns #Used with matplotlib to create the scatter plot

#My input texts
my_inputs = ["On 28th February, a food poisoning incident was reported at Ben's in suburban Bengaluru.",
             "The incident was reported to the Bengaluru food ministry."]

#Random input texts
random_input = ["Preliminary findings suggest that the source of the contamination of food may have been rat poison and poor quality of raw food.",
                "Several people were hospitalized due to this food poisoning incident at the nearest hospital.",
                "Shankar Mahadevan is going to perform at VIT Vellore.",
                "Symptoms reported by affected individuals included nausea, vomiting, and diarrhea.",
                "Please transfer Rs. 10000 to my account, I'll return it later.",
                "As a result of the incident, 83 individuals sought medical attention, with 32 requiring hospitalization.",
                "The milk expenses for the month is Rs. 3000",
                "Benedict's Sherlock Holmes is one of the best Amazon Prime series."]

#My input texts
my_input = ["Recommendations were made to improve hygiene practices, and implement regular monitoring procedures.",
            "The food ministry advised people who had dined at Ben's during the affected period and experienced symptoms to seek medical attention."]

data = my_inputs + random_input + my_input #Combining all the texts

vectorizer = TfidfVectorizer() #Vectorizing the data
X = vectorizer.fit_transform(data)
k = 2 #Number of clusters
kmeans = KMeans(n_clusters=k) #Creating a k-means model and fitting it to the data
kmeans.fit(X)
prediction = kmeans.predict(X) #Predicting the clusters

keyword = "food poisoning" #Keyword for detecting
cluster_with_keyword = None #Finding cluster with the keyword
for i, text in enumerate(data):
    if keyword.lower() in text.lower():
        cluster_with_keyword = prediction[i]
        break

print(prediction) #Printing the cluster assignments (0 or 1) for k=2

#Printing the project brief
if cluster_with_keyword is not None:
    print(f"The {keyword} cluster: {cluster_with_keyword}")
    food_poisoning_texts = [data[i] for i in range(len(data)) if prediction[i] == cluster_with_keyword]
    project_brief = "\n".join(food_poisoning_texts)
    print("Project Brief:")
    print(project_brief)
else:
    print(f"'{keyword}' not found in any cluster.")

X = X.toarray() # Plotting a graph of the data
sns.scatterplot(x=X[:,0], y=X[:,10], hue=prediction)
plt.show()
