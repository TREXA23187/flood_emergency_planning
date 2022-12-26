class ExceptionHandler:
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return f"----------Exception----------\n\t{repr(self.error)}"

    # record the error
    def log(self):
        print(self)
