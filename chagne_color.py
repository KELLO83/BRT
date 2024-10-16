# Function to load and process the image, swap colors, and save the result
from PIL import Image
def swap_colors_in_image(image_path, output_path):
    # Load the original image
    image = Image.open(image_path)

    # Convert the image to editable mode
    image = image.convert("RGBA")

    # Extract data from the image
    data = image.getdata()

    # Define the new color swapping logic
    new_data = []
    for item in data:
        # Swap the gray seats (161, 161, 162) with orange (8, 81, 190)
        if item[0] in range(160, 165) and item[1] in range(160, 165) and item[2] in range(160, 165):
            new_data.append((255, 0, 0, 255))  # Orange color
        # Swap the orange seats (8, 81, 190) with gray (161, 161, 162)
       # elif item[0] in range(180, 190) and item[1] in range(80, 85) and item[2] in range(0, 10):
            #new_data.append((128, 128, 128, 255))  # Gray color


        elif item[0] == 190 and item[1] == 81 and item[2] == 8:
            new_data.append((128, 128, 128, 255))  # Gray color
        else:
            new_data.append(item)
    # Update the image with the new data
    image.putdata(new_data)

    # Save the modified image
    image.save(output_path)
    return output_path

# Call the function with provided input and output paths
image_path = '2번.png'
output_path = 'v3.png'
swap_colors_in_image(image_path, output_path)
