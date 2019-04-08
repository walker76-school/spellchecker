from SpellCorrector import SpellCorrector

if __name__ == "__main__":

    corrector = SpellCorrector()

    print("Done creating SpellChecker")

    wrong = corrector.check("I was aple to sleep tonight. ")
    print(wrong)

    wrong = corrector.check("The water aill is thirty dollars. ")
    print(wrong)

    wrong = corrector.check("The European Southern Observatory will release the first glimpse of a collapsed btar in the center of our galaxy. ")
    print(wrong)

    wrong = corrector.check("Tomorrow is a brand ewnd day.")
    print(wrong)

    wrong = corrector.check("The road lpeds to nowprae.")
    print(wrong)

    wrong = corrector.check("John kicks the uall to the brick uall. ")
    print(wrong)
