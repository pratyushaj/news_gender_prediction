''' Creates regexes for a number of word lists. Word lists may be informative features according to sociolinguistic research on the way male and female leaders are described differently.'''

communal_list = ["affection", "help", "kind", "sympath", "sensitive", "interpersonal[A-Za-z]*", "nurturing", "nurturant", "gentle", "collaborat[A-Za-z]*", "cooperat", "considerat", "sociab", "benevol", "caring", "friendly", "loving", "warm", "enthusias", "passion", "welcom", "outgoing", "assist", "support", "team", "community", "empath"]

agentic_list = ["aggressive", "ambitious", "dominant", "forceful", "independent", "daring", "confident", "competitive", "effective", "drive", "drove", "deliver", "results", "initiate", "initiative", "bold", "execute", "execution", "driving", "achieve", "build", "built", "develop", "strategize", "plan"]

grindstone_list = ["hardworking", "conscientious", "depend[A-Za-z]*", "meticulous", "thorough", "diligen[A-Za-z]*", "dedicat[A-Za-z]*", "careful", "reliab[A-Za-z]*", "effort[A-Za-z]*", "assiduous", "trust[A-Za-z]*", "responsib[A-Za-z]*", "methodical", "industrious", "busy", "work[A-Za-z]*", "persist[A-Za-z]*", "organiz[A-Za-z]*", "discipline[A-Za-z]*", "attentive", "painstaking", "tireless"]

genius_list = ["genius", "vision", "visionary", "champion", "game-?changer", "expert", "master", "gifted", "star", "brilliant", "intelligent", "innovat[A-Za-z]*"]"

standout_list = ["unique", "excellent", "superb", "outstanding", "exceptional", "unparalleled", "finest", "[A-Za-z]*est", "most", "wonderful", "terrific", "fabulous", "magnificent", "marvelous", "remarkable", "extraordinar[A-Za-z]*", "amazing", "supreme[A-Za-z]*", "unmatched", "unrivaled", "super", "superior", "matchless", "optimal", "brilliant", "distinguished", "masterful", "impressive", "exceptional", "prominent", "flawless", "ideal", "impeccable"]

HBR_men_pos = ["analytical", "competent", "athletic", "dependable", "confident", "versatile", "articulate", "level-?headed", "logical", "practical"]

HBR_women_pos = ["compassionate", "enthusiastic", "energetic", "organized"]

HBR_men_neg = ["arrogant", "irresponsible"]

HBR_women_neg = ["inept", "selfish", "frivolous", "passive", "scattered", "opportunistic", "gossip", "excitable", "vain", "panicky", "temperamental", "indecisive"]

socio_regexes = [communal_list, agentic_list, grindstone_list, genius_list, standout_list, HBR_men_pos, HBR_women_pos, HBR_men_neg, HBR_women_neg]

def socio_features(text):
    lists = defaultdict(int)
    for regex in regexes:
		if re.findall(regex, text) != []:
			lists[regex] = 1 #binary categorizer
    return (lists)
