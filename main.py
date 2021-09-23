import networkx as nx
import matplotlib.pyplot as plt
import random
from enum import Enum, auto


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    Modified variant of hierarchy to work with DIgraphs
    '''

    if not isinstance(G, nx.DiGraph):
        raise ValueError("G must be a DiGraph")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        # Place starting node
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            if root in pos:
                pos[root] = (xcenter, min(vert_loc, pos[root][1]))
            else:
                pos[root] = (xcenter, vert_loc)

        children = list(G.neighbors(root))

        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    oldpos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    pos = {}
    for item in oldpos.items():
        if item[1][1] in pos:
            pos[item[1][1]].append([item[0], item[1][0]])
        else:
            pos[item[1][1]] = [[item[0], item[1][0]]]

    for level in pos.items():
        dx = width / len(level[1])
        nextx = xcenter - width / 2 - dx / 2
        for item in level[1]:
            nextx += dx
            oldpos[item[0]] = (nextx, level[0])


    return oldpos


class DecayMode(Enum):
    UN = auto()  # Undefined

    AL = auto()  # Alpha
    A2 = auto()  # 2 alpha
    P1 = auto()  # 1 proton
    P2 = auto()  # 2 proton
    N1 = auto()  # 1 neutron
    N2 = auto()  # 2 neutron
    N4 = auto()  # 4 neutron
    SF = auto()  # Fission
    CD = auto()  # Cluster decay

    BP = auto()  # Beta+ decay
    BM = auto()  # Beta- decay
    EC = auto()  # Electron capture
    B2 = auto()  # Double beta decay (2b-)
    E2 = auto()  # Double electron capture
    Q2 = auto()  # Double positron decay (2b+)

    IT = auto()  # Gamma isomeric
    IC = auto()  # Gamma internal

    EP = auto()  # Electron capture delayed proton
    EN = auto()  # Electron capture delayed neutron
    EF = auto()  # Electron capture delayed fission
    EA = auto()  # Electron capture delayed alpha

    E2P = auto()  # Electron capture delayed double proton
    E3P = auto()  # Electron capture delayed triple proton
    EAP = auto()  # Electron capture delayed proton alpha

    BN = auto()  # Electron delayed neutron
    BF = auto()  # Electron delayed fission
    BA = auto()  # Electron delayed alpha

    B2N = auto()  # Electron delayed double neutron
    B3N = auto()  # Electron delayed triple neutron
    B4N = auto()  # Electron delayed quad neutron

    BNA = auto()  # Electron delayed neutron alpha
    B3A = auto()  # Electron delayed triple alpha
    TBN = auto()  # Double electron delayed neutron


class IngestIsotope:
    def __init__(self, proton_count: int, neutron_count: int, halflife: float, name: str, decaymode, energylevel, jpi, massexc, massunc,
                 abundance, abundunc, branching) -> None:
        self.protons = proton_count
        self.name = name

        self.neutrons = neutron_count
        self.weight = proton_count + neutron_count
        self.halflife = halflife
        self.abundance = abundance
        self.abundunc = abundunc

        self.branching = branching
        self.decaymode = decaymode
        self.energylevel = energylevel
        self.jpi = jpi
        self.massexc = massexc
        self.massunc = massunc
    def get_identifier(self) -> str:
        return self.name + str(self.weight)


class Decay:
    def __init__(self, decaymode: DecayMode, branching: float, energylevel: float) -> None:
        self.branching = branching
        self.decaymode = decaymode
        self.energylevel = energylevel
        self.resultant_isotope = None
        self.is_valid = True


class Isotope:
    def __init__(self, proton_count: int = 0, name: str = None, neutron_count: int = -1, halflife: float = None, jpi=None,
                 abundance: float = None, abundunc: int = None, massexc: float = None, massunc: int = None, decays: list = None) -> None:
        self.protons = proton_count
        self.name = name

        self.neutrons = neutron_count
        self.weight = proton_count + neutron_count
        self.halflife = halflife
        self.abundance = abundance
        self.abundunc = abundunc
        self.jpi = jpi
        self.massexc = massexc
        self.massunc = massunc

        self.decays = decays

    def init_isotope(self, isotope: IngestIsotope):
        self.protons = isotope.protons
        self.name = isotope.name

        self.neutrons = isotope.neutrons
        self.weight = isotope.protons + isotope.neutrons
        self.halflife = isotope.halflife
        self.abundance = isotope.abundance
        self.abundunc = isotope.abundunc

        self.jpi = isotope.jpi
        self.massexc = isotope.massexc
        self.massunc = isotope.massunc

        self.decays = [Decay(isotope.decaymode, isotope.branching, isotope.energylevel)]

    def add_decay_from_ingest(self, isotope: IngestIsotope):
        self.decays.append(Decay(isotope.decaymode, isotope.branching, isotope.energylevel))

    def get_identifier(self) -> str:
        return self.name + str(self.weight)


def load_isotope_data() -> list:
    file = open("Element info.txt")
    file = file.read()
    file = file.split("\n")

    isotopes = []
    for i, line in enumerate(file[1:]):
        line = [line[:4], line[4:12], line[12:16], line[16:20], line[20:29], line[29:44], line[44:65], line[65:76], line[76:102], line[102:128],
                line[128:139], line[139:148], line[148:157], line[157:]]

        # Atomic number int
        line[0] = int(line[0])

        # Element name 2 chars
        line[1] = line[1][:2]

        # Proton count int
        line[2] = int(line[2])

        # Neutron count int
        line[3] = int(line[3])

        # Energy float. -1 -> undef
        if line[4] == "        	":
            line[4] = -1
        line[4] = float(line[4])

        # Jpi data unprocessed
        line[5] = line[5][:-1].rstrip()
        line[5]

        # Mass excess float. infinity -> undef
        if line[6] == "                    	":
            line[6] = float("inf")
        line[6] = float(line[6])

        # Mass uncertainty int. -1 -> undef, -2 -> AP tag
        if line[7] == "          	":
            line[7] = -1
        elif line[7] == "AP        	":
            line[7] = -2
        line[7] = int(line[7])

        # Half life text unprocessed
        line[8]
        # Half life in seconds unprocessed
        line[9]
        if line[9] == "                         	":
            line[9] = float(-1)
        else:
            line[9] = float(line[9])

        # Abundance unprocessed
        if line[10] == "          	":
            line[10] = "-1%"
        line[10] = line[10].strip()[:-1]
        line[10] = float(line[10])

        # Uncertainty int, 1 -> undef
        if line[11] == "        	":
            line[11] = -1
        line[11] = int(line[11])

        # Decay mode decay_class or string
        line[12] = line[12][:-1].rstrip()
        if not line[12]:
            line[12] = DecayMode.UN
        elif line[12] == "A":
            line[12] = DecayMode.AL
        elif line[12] == "SF":
            line[12] = DecayMode.SF
        elif line[12] == "IT":
            line[12] = DecayMode.IT
        elif line[12] == "EC":
            line[12] = DecayMode.EC
        elif line[12] == "B-":
            line[12] = DecayMode.BM
        elif line[12] == "EF":
            line[12] = DecayMode.IC
        elif line[12] == "P":
            line[12] = DecayMode.P1
        elif line[12] == "N":
            line[12] = DecayMode.N1
        elif line[12] == "2P":
            line[12] = DecayMode.P2
        elif line[12] == "EP":
            line[12] = DecayMode.EP
        elif line[12] == "EN":
            line[12] = DecayMode.EN
        elif line[12] == "EA":
            line[12] = DecayMode.EA
        elif line[12] == "EF":
            line[12] = DecayMode.EF
        elif line[12] == "BF":
            line[12] = DecayMode.BF
        elif line[12] == "BN":
            line[12] = DecayMode.BN
        elif line[12] == "BA":
            line[12] = DecayMode.BA
        elif line[12] == "2B-":
            line[12] = DecayMode.B2
        elif line[12] == "2EC":
            line[12] = DecayMode.E2
        elif line[12] == "2N":
            line[12] = DecayMode.N2
        elif line[12] == "2A":
            line[12] = DecayMode.A2
        elif line[12] == "4N":
            line[12] = DecayMode.N4
        elif line[12] == "B2N":
            line[12] = DecayMode.B2N
        elif line[12] == "B3N":
            line[12] = DecayMode.B3N
        elif line[12] == "B4N":
            line[12] = DecayMode.B4N
        elif line[12] == "BNA":
            line[12] = DecayMode.BNA
        elif line[12] == "B3A":
            line[12] = DecayMode.B3A
        elif line[12] == "E2P":
            line[12] = DecayMode.E2P
        elif line[12] == "E2":
            line[12] = DecayMode.E2
        elif line[12] == "EAP":
            line[12] = DecayMode.EAP
        elif line[12] == "E3P":
            line[12] = DecayMode.E3P
        elif line[12] == "2BN":
            line[12] = DecayMode.TBN
        elif line[12] == "B2":
            line[12] = DecayMode.B2

        # Element release
        elif "".join([i for i in line[12] if i not in [str(i) for i in range(10)]]) in ["Si", "Ne", "Mg", "O", "C"]:
            line[12] = line[12].strip()

        # Branching float, -1 -> undef or ?, -2 -> N tag. Ignores bounding info, approximations and &
        line[13] = line[13].rstrip()
        if not line[13]:
            line[13] = -1
        elif "?" in line[13]:
            line[13] = line[13][1:]
            if not line[13]:
                line[13] = -1
        elif "<" in line[13]:
            if "=" in line[13]:
                line[13] = line[13][1:]
            line[13] = line[13][1:]
        elif ">" in line[13]:
            if "=" in line[13]:
                line[13] = line[13][1:]
            line[13] = line[13][1:]
        elif "~" in line[13]:
            line[13] = line[13][1:]
        elif "P" in line[13]:
            line[13] = line[13][1:]
        elif "N" in line[13]:
            line[13] = line[13][1:]
            if not line[13]:
                line[13] = -2
        elif "&" in line[13]:
            line[13] = line[13][1:]
        if "E" in str(line[13]):
            line[13] = line[13].split("E")
            line[13] = float(line[13][0]) * 10 ** int(line[13][1])

        line[13] = float(line[13])

        isotopes.append(IngestIsotope(line[2], line[3], line[9], line[1], line[12], line[4], line[5], line[6], line[7], line[10], line[11], line[13]))
    return isotopes


def patch_isotope_data(isotope_data) -> dict:
    # Halflife undef = 1.01E-40,0
    # Value with *,0 gives a bounded value instead of exact

    patch_halflife = open("halflife patches.txt").read().split("\n")
    halflife_dict = {}
    for line in patch_halflife:
        line = line.split(",")
        if not line[2]:
            continue
        if line[2] == "1.01E-40" and line[3] == "0":
            continue
        line[1] = int(line[1])
        line[2] = float(line[2])
        line[3] = int(line[3])
        halflife_dict[line[0] + str(line[1])] = (line[2], line[3])

    patch_decay = open("decay patches.txt").read().split("\n")
    decay_dict = {}
    for line in patch_decay:
        line = line.split(",")
        if len(line) == 4:
            continue
        if line[2] == "1.01E-40":
            continue
        line[1] = int(line[1])
        if line[2] == "UN":
            line[2] = DecayMode.UN
        elif line[2] == "A":
            line[2] = DecayMode.AL
        elif line[2] == "SF":
            line[2] = DecayMode.SF
        elif line[2] == "IT":
            line[2] = DecayMode.IT
        elif line[2] == "EC":
            line[2] = DecayMode.EC
        elif line[2] == "B-":
            line[2] = DecayMode.BM
        elif line[2] == "B+":
            line[2] = DecayMode.BP
        elif line[2] == "EF":
            line[2] = DecayMode.IC
        elif line[2] == "P":
            line[2] = DecayMode.P1
        elif line[2] == "N":
            line[2] = DecayMode.N1
        elif line[2] == "2P":
            line[2] = DecayMode.P2
        elif line[2] == "EP":
            line[2] = DecayMode.EP
        elif line[2] == "EN":
            line[2] = DecayMode.EN
        elif line[2] == "EA":
            line[2] = DecayMode.EA
        elif line[2] == "EF":
            line[2] = DecayMode.EF
        elif line[2] == "BF":
            line[2] = DecayMode.BF
        elif line[2] == "BN":
            line[2] = DecayMode.BN
        elif line[2] == "BA":
            line[2] = DecayMode.BA
        elif line[2] == "2B-":
            line[2] = DecayMode.B2
        elif line[2] == "2EC":
            line[2] = DecayMode.E2
        elif line[2] == "2N":
            line[2] = DecayMode.N2
        elif line[2] == "2A":
            line[2] = DecayMode.A2
        elif line[2] == "4N":
            line[2] = DecayMode.N4
        elif line[2] == "B2N":
            line[2] = DecayMode.B2N
        elif line[2] == "B3N":
            line[2] = DecayMode.B3N
        elif line[2] == "B4N":
            line[2] = DecayMode.B4N
        elif line[2] == "BNA":
            line[2] = DecayMode.BNA
        elif line[2] == "B3A":
            line[2] = DecayMode.B3A
        elif line[2] == "E2P":
            line[2] = DecayMode.E2P
        elif line[2] == "E2":
            line[2] = DecayMode.E2
        elif line[2] == "EAP":
            line[2] = DecayMode.EAP
        elif line[2] == "E3P":
            line[2] = DecayMode.E3P
        elif line[2] == "2BN":
            line[2] = DecayMode.TBN
        elif line[2] == "B2":
            line[2] = DecayMode.B2
        if decay_dict.get(line[0] + str(line[1])) is None:
            decay_dict[line[0] + str(line[1])] = line[2]
        else:
            if type(decay_dict[line[0] + str(line[1])]) == DecayMode:
                decay_dict[line[0] + str(line[1])] = [decay_dict[line[0] + str(line[1])], line[2]]
            else:
                decay_dict[line[0] + str(line[1])].append(line[2])

    extra_isotopes = []
    for isotope in isotope_data:
        if isotope.halflife == -1:
            patch_val = halflife_dict.get(isotope.name + str(isotope.weight))
            if patch_val is not None:
                isotope.halflife = float(patch_val[0])
        patch_val = decay_dict.get(isotope.name + str(isotope.weight))
        if patch_val is not None:
            if type(patch_val) == DecayMode:
                isotope.decaymode = patch_val
            elif type(patch_val) == list:
                isotope.decaymode = patch_val[0]
                for patched_decay in patch_val[1:]:
                    extra_isotopes.append(IngestIsotope(isotope.protons, isotope.neutrons, isotope.halflife, isotope.name, patched_decay,
                                                        isotope.energylevel, isotope.jpi, isotope.massexc, isotope.massunc,
                                                        isotope.abundance, isotope.abundunc, isotope.branching))
    return isotope_data + extra_isotopes


def anneal_isotopes(isotope_data: list) -> dict:
    annealed_isotopes = {}
    for isotope in isotope_data:
        isotope_listing = annealed_isotopes.get(isotope.get_identifier())
        if isotope_listing is None:
            temp_iso = Isotope()
            temp_iso.init_isotope(isotope)
            annealed_isotopes[isotope.get_identifier()] = temp_iso
        else:
            isotope_listing.add_decay_from_ingest(isotope)
            annealed_isotopes[isotope.get_identifier()] = isotope_listing
    return annealed_isotopes


def parse_target(target) -> str:
    letters = [chr(i) for i in range(65, 91)]
    numbers = [str(i) for i in range(10)]
    lettarget = "".join([char.upper() for char in target if char.upper() in letters]).capitalize()
    lettarget += " " * (2 - len(lettarget))
    numtarget = "".join([char.upper() for char in target if char in numbers])
    return "".join(lettarget + numtarget)


def compute_links(isotope_data):
    element_names = ["xx", 'H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar', 'K ', 'Ca', 'Sc',
                     'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo',
                     'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
                     'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
                     'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db',
                     'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', "Og"]

    transition_constants = {'A2': (-4, -4), 'AL': (-2, -2), 'B2': (2, -2), 'B2N': (1, -3), 'B3A': (-5, -7), 'B3N': (1, -4), 'B4N': (1, -5),
                            'BA': (-1, -3), 'BF': 'ffff', 'BM': (1, -1), 'BN': (1, -2), 'BNA': (-1, -3), 'BP': (0, -1), 'E2': (-2, 2),
                            'E2P': (-3, 1), 'E3P': (-4, 1), 'EA': (-3, -1), 'EAP': (-4, -1), 'EC': (-1, 1), 'EF': 'ffff', 'EN': (-1, 0),
                            'EP': (-2, 1), 'IC': (0, 0), 'IT': (0, 0), 'N1': (0, -1), 'N2': (0, -2), 'N4': (0, -4), 'P1': (-1, 0), 'P2': (-2, 0),
                            'Q2': (-2, 2), 'SF': 'ffff', 'TBN': (2, -3), 'UN': '----'}

    for key, iso in isotope_data.items():
        for decay in iso.decays:
            if type(decay.decaymode) == DecayMode:
                decay_const = transition_constants[decay.decaymode.name]
                if decay_const is None:
                    raise LookupError(f"No decay constants were defined for a decay mode of {key}")
                if decay_const == "ffff":
                    decay.resultant_isotope = "ER01"  # Mark isotope as invalid, fission type unsupported
                    decay.is_valid = False
                elif decay_const == "----":
                    decay.resultant_isotope = "ER00"  # Mark isotope as invalid, unknown decay mode
                    decay.is_valid = False
                else:
                    new_key = [iso.protons + decay_const[0], iso.neutrons + decay_const[1]]
                    new_key = [element_names[new_key[0]], str(new_key[0] + new_key[1])]
                    new_key = "".join(new_key)
                    decay.resultant_isotope = new_key
            else:
                decay_const = parse_target(decay.decaymode)
                if len(decay_const) != 4:
                    raise ValueError(f"Decay product {decay_const} is invalid or incomplete for isotope {key}")

                new_key = [decay_const[:2], decay_const[2:]]  # Convert to proton and neutron count
                new_key[0] = element_names.index(new_key[0])
                new_key[1] = int(new_key[1]) - new_key[0]

                new_key = [iso.protons - new_key[0], iso.neutrons - new_key[1]]  # Apply to original
                new_key = [element_names[new_key[0]], str(new_key[0] + new_key[1])]
                new_key = "".join(new_key)
                decay.resultant_isotope = new_key

    return isotope_data


def print_data_errors(isotope_data) -> None:
    parsed_list = []
    for isotope in isotope_data.values():
        sum = 0
        if isotope.halflife == float("inf"):
            continue
        for decay in isotope.decays:
            sum += decay.branching
        sum = round(sum)
        if sum <= 0:
            parsed_list.append(isotope.get_identifier())
    print(f"{len(parsed_list)} items are unstable and have missing branching factors.")
    print(parsed_list)
    print()

    parsed_list = []
    for isotope in isotope_data.values():
        sum = 0
        if isotope.halflife == float("inf"):
            continue
        for decay in isotope.decays:
            sum += decay.branching
        sum = round(sum)
        if sum != 100 and sum > 0:
            parsed_list.append([isotope.get_identifier(), sum])
    print(f"{len(parsed_list)} items are unstable and have branching factors that do not add to 100.")
    print(parsed_list)
    print()

    parsed_list = []
    for isotope in isotope_data.values():
        if isotope.halflife == -1:
            parsed_list.append(isotope.get_identifier())
    print(f"{len(parsed_list)} items have no given halflife")
    print(parsed_list)
    print()


def describe(isodat, target: str, detail="Minimal") -> None:
    print()
    target = parse_target(target)
    iso = isodat.get(target)
    if iso is None:
        print(f"The requested isotope, {target}, does not exist")
    else:
        print(f"{target} has {iso.protons} protons and {iso.neutrons} neutrons. " +
              f"It has a half life of {'{:.4e}'.format(round(iso.halflife))} seconds and {len(iso.decays)} known decay paths.")
        if detail.lower() == "advanced":
            print()
            print(f"The decay paths of {target} are as follows:")
            for decay in iso.decays:
                if type(decay.decaymode) == str:
                    print(f"Spontaneous fission into {decay.decaymode} with a probability of {decay.branching}%")
                else:
                    print(f"{decay.decaymode.name} decay with a probability of {decay.branching}%")
        print()


def decay(isodat, base: str):
    base = parse_target(base)
    if isodat.get(base) is None:
        raise ValueError(f"Given isotope {base} does not exist")

    g = nx.DiGraph()
    g.add_node(base)
    open_nodes = [base]
    labels = {}

    while open_nodes:
        current_iso = open_nodes.pop(0)
        iso_listing = isodat.get(current_iso)
        if iso_listing is None:
            print(current_iso, decay_type.decaymode)
            continue
        for decay_type in iso_listing.decays:
            if decay_type.is_valid:
                if decay_type.decaymode not in [DecayMode.IT, DecayMode.IC]:
                    open_nodes.append(decay_type.resultant_isotope)
                    iso_listing = isodat.get(decay_type.resultant_isotope)
                    if iso_listing is None:
                        print(current_iso, decay_type.decaymode)
                        continue                                                                                    
                    g.add_edge(current_iso, decay_type.resultant_isotope, weight=max(decay_type.branching, 2)/50)
                    if type(decay_type.decaymode) == DecayMode:
                        labels[(current_iso, decay_type.resultant_isotope)] = decay_type.decaymode.name
                    else:
                        labels[(current_iso, decay_type.resultant_isotope)] = decay_type.decaymode

    pos = hierarchy_pos(g, base, vert_gap=0.2)
    plt.figure()
    edge_weights = [g[u][v]['weight'] for u, v in g.edges()]
    nx.draw_networkx(g, pos, node_size=500, node_shape="s", font_size=6, width=edge_weights)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, font_color='red', font_size=6)
    plt.axis("off")
    plt.fill()
    plt.title(f"Decay chain of {base}")
    plt.show()


def main() -> None:
    isodat = load_isotope_data()
    isodat = patch_isotope_data(isodat)
    isodat = anneal_isotopes(isodat)
    isodat = compute_links(isodat)

    # print_data_errors(isodat)

    # isodat is a dictionary of isotopes with keys NNwww where N is name and w is atomic weight
    # Each isotope holds a set of possible decays

    # Planned functions:
    # print_data_errors
    # describe(isodat, "U235", "advanced")
    # Cool ones: Th212, U238
    # decay(isodat, "Th212")
    # Revert
    # Find
    print("Welcome to the atomic decay sim.")
    print("Type a command below, followed by the isotope requested, or type help")
    current_input = input(">: ").lower()
    while current_input not in ["quit", "exit", "q", "e", "break", "b"]:
        if current_input in ["help", "h", "?"]:
            print("This program visualises the decay chain of a given isotope as well as containing various search tools")
            print()
            print("  Use the command decay followed by an isotope to show a decay tree. Eg: decay 212TH or decay u-238")
            print("  Use the command describe followed by an isotope to see a description of the isotope. For more info, include a word from (simple,"
                  " advanced, full). Eg: describe th 212 advanced or describe full 238-u")
            print("  Use the errors command to print all known errors and ommisions in the database")
        else:
            new_input = current_input.split()
            if new_input[0] in ["decay"]:
                if len(new_input) > 2:
                    print(f"Your command '{current_input}' has too many arguments. Please only provide the isotope requested")
                elif len(new_input) < 2:
                    print(f"Your command '{current_input}' has too few arguments. Please only provide the isotope requested")
                else:
                    decay(isodat, new_input[1])
            elif new_input[0] in ["errors", "error", "err", "problems", "problem"]:
                if len(new_input) > 1:
                    print(f"Your command '{current_input}' has too many arguments. Please do not provide other arguments")
                else:
                    print_data_errors(isodat)
            elif new_input[0] in ["describe", "explain", "whatis"]:
                if len(new_input) > 3:
                    print(f"Your command '{current_input}' has too many arguments. Please provide the isotope requested and optionally a detail word")
                elif len(new_input) < 2:
                    print(f"Your command '{current_input}' has too few arguments. Please provide the isotope requested and optionally a detail word")
                else:
                    if len(new_input) == 2:
                        new_input.append("simple")
                    describe(isodat, new_input[1], new_input[2])
            else:
                print(f"{current_input} was not recognised. Type help for help with usage")

        current_input = input(">: ").lower()


if __name__ == "__main__":
    main()
