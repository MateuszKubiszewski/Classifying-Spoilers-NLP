# Classifying-Spoilers-NLP

To install the environment for this project poetry package is needed.

`pip install poetry`

Then inside the folder directory:

`poetry install`

To install needed packages. To run a script from within environment:

`poetry shell`
`python <script-name>`

To add a package to the project you have to run:

`poetry add <package-name>`

If somebody adds a package, then after pulling the lates version of repository we can use poetry install once again to install new dependencies.

Directory tira contains example Dockerfile with needed python script, for creating image of solution. It doesn't contain model and tokenizer, wchich are also needed for creating proper image, but due to their sizes were not included. They can be downloaded from: https://drive.google.com/drive/folders/1c2WHoRJV73RrvzTOSWzLkReZ7cccaxbA?usp=sharing.
