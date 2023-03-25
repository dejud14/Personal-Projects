"""
Daniel Dejud
All work is original.
"""
import pandas
import openpyxl
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
# First, we need to normalize the experimental data with respect to the control data.
def normalize(controlFilename, experimentalFilename):
    """
    The normalize() function computes the ratio of every cell in the experimental Excel file with
    every cell in the control Excel file. It does so by utilizing the "pandas" module to read in
    the values from the Excel files into DataFrames and then calculating the ratio between them.

    Arguments: 
        controlFilename (string):       the filename for the Excel document where the control quantities are;
                                        it has format <filename>.<file extension>
        experimentalFilename (string):  the filename for the Excel focument where the experimental quantities are;
                                        it has format <filename>.<file extension>

    Returns: 
        normalizedData (DataFrame): Dataframe containing the ratio between the experimental DataFrame and the 
                                    control DataFrame.
        experimental (DataFrame):   DataFrame containing the experimental data.
    """
    # Read data from the provided control file.
    control = pandas.read_excel(controlFilename, header = None)

    # Read data from the provided experimental file.
    experimental = pandas.read_excel(experimentalFilename, header = None)

    # Normalize and send back the wanted normalized data.
    normalizedData = experimental / control

    return normalizedData, experimental


# We determine an addequate threshold to analyze which values will be taken into account.
def findThreshold(data, multiplier=1.0, nFolds=5):
    """
    The findThreshold() function determines an adequate threshold based on the relationship between
    experimental data and the ratio of experimental to control data. It uses a linear regression
    model to learn this relationship and calculates the threshold based on the root mean squared
    error (RMSE) obtained from cross-validation.

    Arguments:
        data (DataFrame): 	A DataFrame containing two columns: 'Experimental' and 'Ratio'.
                            'Experimental' contains experimental data values,
                            and 'Ratio' contains the corresponding ratio of experimental to control data.
        multiplier (float): A multiplier to adjust the calculated threshold value. Default is 1.0.
        nFolds (int):       The number of folds to use in cross-validation. Default is 5.

    Returns:
        threshold (float): The calculated threshold value based on the RMSE and the given multiplier.
    """
    
    # Prepare the input features and target variable.
    x = data['Experimental'].values.reshape(-1, 1)
    y = data['Ratio'].values.reshape(-1, 1)

    # Create a linear regression model.
    model = LinearRegression()

    # Create a KFold cross-validator object with the specified number of folds.
    kfold = KFold(n_splits=nFolds, shuffle=True, random_state=42)

    # Calculate the cross-validated RMSE scores.
    rmseScores = cross_val_score(model, x, y, scoring='neg_root_mean_squared_error', cv=kfold)

    # Calculate the average RMSE score.
    avgRmse = -1 * rmseScores.mean()

    # Calculate the threshold using the given multiplier and the average RMSE score.
    threshold = avgRmse * multiplier

    return threshold


# We apply the threshold and filter the values, separating anomalies.
def threshold(normalizedData, experimental, multiplier=1.0, nFolds = 5):
    """
    The threshold() function selects values from the normalizedData DataFrame that are within
    an acceptable threshold. The threshold is calculated based on the relationship between
    experimental data and the ratio of experimental to control data.

    Arguments:
        normalizedData (DataFrame): A DataFrame containing the normalized data (ratio of
                                    experimental to control data).
        experimental (DataFrame):   A DataFrame containing the experimental data.
        multiplier (float):         A multiplier to adjust the calculated threshold value. Default is 1.0.
        nFolds (integer):           The number of folds to use in cross-validation. Default is 5.

    Returns:
        withinthreshold (list): A list of tuples containing the accepted values and their
                                original cell locations in the experimental data.
    """
    
    withinthreshold = []

    # Create a DataFrame containing experimental values and their corresponding ratios.
    data = pandas.DataFrame({"Experimental": experimental.values.flatten(), "Ratio": normalizedData.values.flatten()})
    
    # Calculate the acceptable threshold using the findThreshold function.
    acceptableThreshold = findThreshold(data, multiplier, nFolds)

    # Set the lower and upper bounds based on the acceptable threshold.
    lowerBound = 1 - acceptableThreshold
    upperBound = 1 + acceptableThreshold

    # Iterate through the normalized data and select values within the threshold bounds.
    for row in range(normalizedData.shape[0]):
        for column in range(normalizedData.shape[1]):
            value = normalizedData.iloc[row, column]
            saveValue = experimental.iloc[row, column]
            # Filter and record the anomalies
            if value <= lowerBound or value >= upperBound:
                originalLocation = f"{openpyxl.utils.get_column_letter(column + 1)}{row + 1}"
                withinthreshold.append((saveValue, originalLocation))
    return withinthreshold

# 
def outputFile(withinthreshold, outputFilename):
    """
    The outputFile() function writes the accepted values and their original cell locations
    to a CSV file.

    Arguments:
        withinthreshold (list): A list of tuples containing the accepted values and their
                                original cell locations in the experimental data.
        outputFilename (str): 	The filename for the output CSV file. It has format
                              	<filename>.<file extension>.
    """
    
    # Create a DataFrame from the withinthreshold list and write it to a CSV file.
    df = pandas.DataFrame(withinthreshold, columns=["Value", "Cell"])
    df.to_csv(outputFilename, index=False)

def getMultiplierAndFolds():
    """
    The getMultiplierAndFolds() function asks the user if they have a specified threshold multiplier
    and the number of folds for cross-validation. If the user inputs 'y' for the multiplier,
    it prompts the user to input the multiplier and processes the files using the specified
    multiplier. If the user inputs 'n', it processes the files using the default multiplier (1.0).

    The function also asks the user if they have a specified number of folds for cross-validation.
    If the user inputs 'y', it prompts the user to input the number of folds and processes the
    files using the specified number of folds. If the user inputs 'n', it processes the files
    using the default number of folds (5).

    The function recursively calls itself in case the user provides an invalid input.
    """
    # Ask the user if they have a specified threshold multiplier
    inqMultiplier = input("Do you have a specified threshold multiplier? (y/n) ")

    if inqMultiplier == "y":
        # Prompt the user to input the multiplier and convert it to float
        multiplier = float(input("Enter the multiplier (decimal number): "))
    elif inqMultiplier == "n":
        multiplier = 1.0
    else:
        # Handle invalid input and retry
        print("Invalid input, please try again.")
        getMultiplierAndFolds()
        return

    # Ask the user if they have a specified number of folds for cross-validation
    inqFolds = input("Do you have a specified number of folds for cross-validation? (y/n) ")

    if inqFolds == "y":
        # Prompt the user to input the number of folds and convert it to int
        numFolds = int(input("Enter the number of folds (integer): "))
    elif inqFolds == "n":
        numFolds = 5
    else:
        # Handle invalid input and retry
        print("Invalid input, please try again.")
        getMultiplierAndFolds()
        return

    # Process the files using the specified multiplier and number of folds
    for (controlFile, experimentalFile), outputName in zip(inputFiles, outputFilenames):
        processFiles(controlFile, experimentalFile, outputName, multiplier, numFolds)



def processFiles(controlFile, experimentalFile, outFile, multiplier=1.0, nFolds=5):
    """
    The processFiles() function processes the provided control and experimental files,
    normalizes the data, applies a threshold to identify anomalies, and writes the
    results to an output file.

    Arguments:
        controlFile (str):         The filename for the control Excel file. It has format
                                   <filename>.<file extension>.
        experimentalFile (str):    The filename for the experimental Excel file. It has format
                                   <filename>.<file extension>.
        outFile (str):             The filename for the output CSV file. It has format
                                   <filename>.<file extension>.
        multiplier (float):        A multiplier to adjust the threshold value for the
                                   threshold() function. Default is 1.0.
        nFolds (int):              The number of folds to use in cross-validation in the 
                                   threshold() function. Default is 5.

    """
    # Normalize the data using the provided control and experimental files
    normalizedData, experimentalData = normalize(controlFile, experimentalFile)

    # Apply a threshold to identify anomalies
    withinThreshold = threshold(normalizedData, experimentalData, multiplier, nFolds)

    # Write the results to an output file
    outputFile(withinThreshold, outFile)

"""
Please enter the control and experimental filenames (with file extension), respectively,
encased in between quotation marks. The same applies for the output filenames.
All the files need to be in the same folder as the "Daniel_Dejud_Data_Science.py" file,
and this file should be executed on that folder.
Please take below as an example.
"""
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change your input file filenames here!
    inputFiles = [
        (os.path.join(script_dir, "Exam_control measurements1.xlsx"), os.path.join(script_dir, "Exam_experimental_measurements1.xlsx")),
        (os.path.join(script_dir, "Exam_control measurements2.xlsx"), os.path.join(script_dir, "Exam_experimental_measurements2.xlsx"))
    ]
    # Change your output file filenames here!
    outputFilenames = [
        os.path.join(script_dir, "Experiment 1 Anomalies.csv"),
        os.path.join(script_dir, "Experiment 2 Anomalies.csv")
    ]
    
    try:
        getMultiplierAndFolds()
    except FileNotFoundError:
        print("Sorry, the specified file cannot be found. Fix the filename and try again.")
    except ValueError:
        print("Your input files do not follow the format. Please fix this.")

