
import pathlib
for path in pathlib.Path("data/images/unofficial").iterdir():
    if path.is_file():
        old_name = path.stem
        old_extension = path.suffix
        directory = path.parent

        #If path is less than 6 chars long, add leading 0's
        original_length = len(old_name)
        if(original_length < 6):
            leading_zero_string = "0" * (6 - original_length)

        new_name = leading_zero_string + old_name + old_extension #create the new name
        path.rename(pathlib.Path(directory, new_name)) #set the new name