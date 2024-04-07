# AI Chat with Your Files

## Introduction
This Streamlit application allows users to upload PDF or Word files and interact with the content through a chat interface. By leveraging the power of Retrieval-Augmented Generation (RAG) and open-source language models (LLMs), users can ask questions about the uploaded document and receive insightful answers, making it easier to extract information and gain knowledge from their files.

## Features
- Upload PDF or Word files.
- Ask questions about the content of the uploaded file.
- Receive answers using RAG and open-source LLMs.
- User-friendly chat interface for interaction.

## Installation

To run this application, you need Python installed on your system. Follow these steps to set up the project:

1. Clone the repository:
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/wuheison/chat_with_files.git)
2. Create and activate a virtual environment (optional but recommended):
python -m venv venv

For Windows
venv\Scripts\activate

For Unix or MacOS
source venv/bin/activate


3. Install the required dependencies:
pip install -r requirements.txt


## Configuration

### Hugging Face Token

To use this application, you need a Hugging Face API token. Follow these steps to configure your token:

1. Create a `.env` file in the root directory of the project.
2. Add your Hugging Face API token to the `.env` file as follows:
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here

   Replace `your_huggingface_api_token_here` with your actual Hugging Face API token.

### Using AWS Secrets Manager (Optional)

If you prefer to use AWS Secrets Manager to manage your API tokens securely, especially when deploying the application on an EC2 instance, follow these steps:

1. Uncomment the AWS Secrets Manager section in the `main.py` file. This section is marked with the comment `# for aws secrete manager on ec2`.
2. Ensure you have configured your AWS credentials and have the necessary permissions to access the Secrets Manager.
3. Store your Hugging Face API token in AWS Secrets Manager and update the `secret_name` and `region_name` variables in the `get_aws_secret()` function accordingly.

## Usage

To start the Streamlit application, run the following command in your terminal:

bash streamlit run main.py


### How to Use the Application

1. **Upload a File**: Drag and drop or click on the "Browse files" button to upload a PDF or Word file.
2. **Wait for Processing**: The application will process the uploaded file to prepare it for question-answering.
3. **Ask Questions**: Type your question in the chat input box.
4. **Receive Answers**: Get answers based on the content of your uploaded file.
5. **Refresh to Restart**: Refresh the page to upload a new file and start a new session.

## Contributing

Contributions to improve the application are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).
