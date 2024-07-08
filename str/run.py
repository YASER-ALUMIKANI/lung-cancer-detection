# with using __init__.py file

from lung_cancer_d import utilities as utils
import warnings
warnings.filterwarnings('ignore')

print(". Loading data.....\n")
print("===================================================================================")
data = utils.load_data()

cond = True
while cond:
    print("""
        1. Loading data by default                  
        2.Top 10 rows                     3. Shape of Dataset            
        4.Data types for each columns     5.descriptive statistics    
        6.distribution of the level       7.Correlations Between Attributes(columns)  
        8.Histograms group data           9.Density plots
        10.Box and whisker plots          11.Rescale the data
        12.standardize the data           13.importance feature of  the data
        14.Choosing the best algorithm    press any key to exit(0)
    ===================================================================================        
    \n""")

    select = input('select your operation from 2 to 14 \n')

    if  select == '1':
        print("\n1. data loading again ")
        print("================================================")

    elif select == '2':
        print("\n2. Top 10 rows ")
        print("================================================")
        utils.deftop_10_rows(data)
    elif select == '3':
        print("\n3. Shape of Dataset ")
        print("================================================")
        utils.shape_data(data)
    elif select == '4':
        print("\n4. Data types for each columns ")
        print("================================================")
        utils.type_data(data)
    elif select == '5':
        print("\n5. descriptive statistics ")
        print("================================================")
        utils.summarize_data(data)
    elif select == '6':
        print("\n6. understand the distribution of the level ")
        print("================================================")
        utils.levle_class(data)
    elif select == '7':
        print("\n7. Correlations Between Attributes(columns) ")
        print("================================================")
        utils.correl(data)
    elif select == '8':
        print("\n8. Histograms group data ")
        print("================================================")
        utils.data_histogram(data)
    elif select == '9':
        print("\n9.   Density plots ")
        print("================================================")
        utils.df_density_plt(data)
    elif select == '10':
        print("\n10.   Box and whisker plots ")
        print("================================================")
        utils.box_whisker_plot(data)
    elif select == '11':
        print("\n11.   Rescale the data ")
        print("================================================")
        arr = data.values
        utils.rescale_data(arr)
    elif select == '12':
        print("\n12.   standardize the data ")
        print("================================================")
        arr = data.values
        utils.standardize_data(arr)
    elif select == '13':
        print("\n13.   importance feature of  the data ")
        print("================================================")
        utils.importince_f(data)
    elif select == '14':
        print("\n14.   Choosing the best algorithm ")
        print("================================================")
        n_data = utils.new_data(data)             #create a new dataframe with the importance features
        utils.convert_to_number(n_data)             #convere string values to numbers


        ml_models = utils.all_models()

        arr = n_data.values

        X = n_data.iloc[:, 0:-1].values

        y = n_data.iloc[:, -1].values
        utils.train_models_scores(ml_models, X, y)
    else:
        cond = False




