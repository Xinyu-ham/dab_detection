import random

def get_random_insults():
    insults = [
        'Why you dabbin\'?',
        'You should be ashamed of youself!',
        'STOP!',
        'Your parents never liked you!',
        'Stop using outdated memes.'
    ]
    return(insults[random.randint(0, 3)])

