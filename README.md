This is the code repository for Jitter-Aware Restoration with Equivalent Jitter Model for Remote Sensing Push-Broom Image. 

**Simple Run Guide: Step-by-Step Instructions (Detailed guide will be updated later)**

1. **Set up the Anaconda + Python environment on Linux**  
   Follow the steps below to configure the necessary environment:

   - Install [Anaconda](https://www.anaconda.com/products/distribution) for managing Python environments.
   - Create a new environment:
     ```bash
     conda create --name <env_name> python=3.8
     conda activate <env_name>
     ```
   
2. **Upload the code files and download the dataset**  
   Upload the code files to your server. Then, execute the following command to download and extract the dataset:
   ```bash
   sh down.sh
   ```
   **Note**: Ensure the server is connected to the internet, or manually download and upload the dataset if needed.

3. **Install necessary Python libraries**  
   Install the required libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate simulation data**  
   - To generate TDI simulation data, run the following:
     ```bash
     python data_generate_TDIv2.py
     ```
   - To generate LA simulation data, run this:
     ```bash
     python data_generate_LinearArray.py
     ```

5. **Set paths for training and validation datasets in Options**  
   Update the `Options` file to point to the paths of your training and validation datasets.

6. **Train the model**  
   Finally, start the training process by running the following command:
   ```bash
   bash run.sh
   ```

**Note:** This is a simple run guide. A more detailed step-by-step instruction will be provided in future updates.
