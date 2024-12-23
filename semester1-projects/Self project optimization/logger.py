import datetime
import sys

class Logger:
    def __init__(self, filename="output", logging="Print"):
        self.logging = logging
        self.filename = filename
        self.log_file = None

        if logging == "TXT":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"log_{self.filename}_{timestamp}.txt"
            self.log_file = open(log_filename, "w")

    def log(self, message):
        if self.logging == "Print":
            print(message)
        elif self.logging == "TXT" and self.log_file:
            self.log_file.write(message + "\n")
        # If logging is "False", do nothing

    def close(self):
        if self.log_file:
            self.log_file.close()

# Example usage of the Logger class
def main():
    logger = Logger(filename="example", logging="TXT")

    logger.log("This is a test log message.")
    logger.log("Logging another message.")

    # Perform operations and log outputs
    for i in range(5):
        logger.log(f"Iteration {i}: Logging some computation result.")

    # Close the logger if a file was created
    logger.close()

if __name__ == "__main__":
    main()
