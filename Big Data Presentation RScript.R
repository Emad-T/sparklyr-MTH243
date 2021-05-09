#All code obtained from spark.rstudio.com for demonstration purposes

#install.packages("sparklyr")      Installing the sparklyr package


#Installing local version of Spark
library(sparklyr)
#spark_install(version = "2.1.0")

#Connecting to local instance of Spark
sc <- spark_connect(master = "local")

#Calling the dplyr library
library(dplyr)


#Installing some datasets to play around with
install.packages(c("nycflights13", "Lahman"))

#spark_read_csv()     Reading a csv file using Spark

#Copying the datasets to Spark
iris_tbl <- copy_to(sc, iris)
flights_tbl <- copy_to(sc, nycflights13::flights, "flights")
batting_tbl <- copy_to(sc, Lahman::Batting, "batting")
mtcars_tbl <- copy_to(sc, mtcars)
highschool_tbl <- copy_to(sc, ggraph::highschool, "highschool")

# %>% is magrittr notation that allows for chaining of functions
# For example, iris %>% head() simply means head(iris)


#---Examples of using dplyr---#

example1<-flights_tbl %>% filter(dep_delay == 2) #Filters the dataset so that it only shows records with departure delays equal to 2
example1

example2<-flights_tbl %>% arrange(desc(dep_delay)) #Arrange dataset by using the departure delays in descending order
example2

example3<-flights_tbl %>% summarise(mean_dep_delay = mean(dep_delay)) #Finding the average of all the departure delays
example3

example4<- flights_tbl %>% select(year:day, arr_delay, dep_delay) #Selecting specific columns from the dataset to view
example4

example5 <- flights_tbl %>% filter(dep_delay == 2)
mutate_example <- mutate(example5, air_time_hours = arr_time / 60) #Creating a new column in the table where the arrival time is divided by 60
mutate_example

example6<-flights_tbl %>%
  filter(month == 5, day == 17, carrier %in% c('UA', 'WN', 'AA', 'DL')) %>%
  select(carrier, dep_delay, air_time, distance) %>%
  arrange(carrier) %>%
  mutate(air_time_hours = air_time / 60) #Combining all the dplyr methods into one example
example6 

#---Machine learning examples with sparklyr---#

#Calling the ggplot2 library
library(ggplot2)

#K-Means Clustering Example
kmeans_model <- iris_tbl %>%
  ml_kmeans(k = 3, features = c("Petal_Length", "Petal_Width"))
kmeans_model

predicted <- ml_predict(kmeans_model, iris_tbl) %>%
  collect
table(predicted$Species, predicted$prediction)

ml_predict(kmeans_model) %>%
  collect() %>%
  ggplot(aes(Petal_Length, Petal_Width)) +
  geom_point(aes(Petal_Width, Petal_Length, col = factor(prediction + 1)),
             size = 2, alpha = 0.5) + 
  geom_point(data = kmeans_model$centers, aes(Petal_Width, Petal_Length),
             col = scales::muted(c("red", "green", "blue")),
             pch = 'x', size = 12) +
  scale_color_discrete(name = "Predicted Cluster",
                       labels = paste("Cluster", 1:3)) +
  labs(
    x = "Petal Length",
    y = "Petal Width",
    title = "K-Means Clustering",
    subtitle = "Use Spark.ML to predict cluster membership with the iris dataset."
  )

#Linear Regression Example
lm_model <- iris_tbl %>%
  select(Petal_Width, Petal_Length) %>%
  ml_linear_regression(Petal_Length ~ Petal_Width)

iris_tbl %>%
  select(Petal_Width, Petal_Length) %>%
  collect %>%
  ggplot(aes(Petal_Length, Petal_Width)) +
  geom_point(aes(Petal_Width, Petal_Length), size = 2, alpha = 0.5) +
  geom_abline(aes(slope = coef(lm_model)[["Petal_Width"]],
                  intercept = coef(lm_model)[["(Intercept)"]]),
              color = "red") +
  labs(
    x = "Petal Width",
    y = "Petal Length",
    title = "Linear Regression: Petal Length ~ Petal Width",
    subtitle = "Use Spark.ML linear regression to predict petal length as a function of petal width."
  )

#Random Forest Example
rf_model <- iris_tbl %>%
  ml_random_forest(Species ~ Petal_Length + Petal_Width, type = "classification")

rf_predict <- sdf_predict(rf_model, iris_tbl) %>%
  ft_string_indexer("Species", "Species_idx") %>%
  collect

table(rf_predict$Species_idx, rf_predict$prediction)

#---Data Visualization with sparklyr---#
library(graphframes)
library(ggraph)
library(igraph)

#Creating vertices
from_tbl <- highschool_tbl %>%
  distinct(from) %>%
  transmute(id = from)

to_tbl <- highschool_tbl %>%
  distinct(to) %>%
  transmute(id = to)


vertices_tbl <- from_tbl %>%
  sdf_bind_rows(to_tbl)

#Creating edges
edges_tbl <- highschool_tbl %>%
  transmute(src = from, dst = to)


#Creating the graphframe
gf_graphframe(vertices_tbl, edges_tbl)


#Visualizing the graphframe
graph <- highschool_tbl %>%
  sample_n(20) %>%
  collect() %>%
  graph_from_data_frame()

ggraph(graph, layout = 'kk') +
  geom_edge_link(aes(colour = factor(year))) +
  geom_node_point() +
  ggtitle('An example')
