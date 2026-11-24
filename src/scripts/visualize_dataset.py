"""Interactive menu with options to visualize generated characters.

From the project root:

    python -m src.scripts.visualize_dataset

"""

import config
from src.tools.dataset_visualizer import DataVisualizer


def display_menu() -> None:
    """Display the available visualization options."""
    print("\n========== Dataset Visualizer ==========")
    print("1. Visualize first N images")
    print("2. Visualize a specific character")
    print("3. List available characters")
    print("0. Exit")
    print("========================================")


def main() -> None:
    """Run the interactive dataset visualization menu."""
    visualizer = DataVisualizer(
        dataset_dir=config.PROCESSED_DIR,
        save_dir=config.PROCESSED_DIR,
    )

    while True:
        display_menu()
        option = input("Select an option: ").strip()

        if option == "1":
            try:
                n = int(input("Number of images: "))
                save = input("Save images? (y/n): ").strip().lower() == "y"

                visualizer.visualize(n=n, save=save)

            except ValueError as e:
                print(f"Invalid input: {e}")

        elif option == "2":
            character = input("Enter character: ").strip()

            try:
                n = int(input("Number of images: "))
                save = input("Save images? (y/n): ").strip().lower() == "y"

                visualizer.visualize(
                    character=character,
                    n=n,
                    save=save,
                )

            except ValueError as e:
                print(f"Invalid input: {e}")

        elif option == "3":
            characters = visualizer.get_characters()

            print(f"\nAvailable characters ({len(characters)}):")
            print(" ".join(characters))

        elif option == "0":
            print("Exiting visualizer...")
            break

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
