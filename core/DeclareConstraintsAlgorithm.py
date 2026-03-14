import numpy as np

ONE_VAR_CONSTRAINTS = ['AtLeastOne', 'AtMostOne', 'Init', 'End']
ONE_VAR_LEN = len(ONE_VAR_CONSTRAINTS)
TWO_VAR_CONSTRAINTS_PARTNERS = [('RespondedExistence', True), ('NotRespondedExistence', True), ('Response', True), ('AlternateResponse', True), ('ChainResponse', True), ('NotResponse', True), ('NotChainResponse', True), ('Precedence', False), ('AlternatePrecedence', False), ('ChainPrecedence', False), ('NotPrecedence', False), ('NotChainPrecedence', False)] # True if the activation is x, False if y
TWO_VAR_CONSTRAINTS = [pair[0] for pair in TWO_VAR_CONSTRAINTS_PARTNERS]
TWO_VAR_LEN = len(TWO_VAR_CONSTRAINTS)

def _YPos(A, Asq, constraint, a, b=None):
    if b is None:
        return ONE_VAR_CONSTRAINTS.index(constraint) * A + a
    else:
        return ONE_VAR_LEN * A + TWO_VAR_CONSTRAINTS.index(constraint) * Asq + a * A + b

def _YPosInv(A, Asq, number):
    b = None
    if number < ONE_VAR_LEN * A:
        constraint = ONE_VAR_CONSTRAINTS[number // A]
        a = number % A
    else:
        newnumber = number - ONE_VAR_LEN * A
        constraint = TWO_VAR_CONSTRAINTS[newnumber // Asq]
        a = (newnumber % Asq) // A
        b = (newnumber % Asq) % A
    return (constraint, a, b)

def _isOneVar(A, Asq, number):
    if number < ONE_VAR_LEN * A:
        return True
    else:
        return False

def _Declare(A, Asq, N_DeclareFeatures, case_i, trace): # case_i is just returned again
    L = len(trace)

    y = np.zeros(N_DeclareFeatures, dtype=np.int8) # global standard: everything zero
    # y has the following format:
    # Example: Assume we have A=10 activities
    # 0-9 is AtLeastOne
    # 10-19 is AtMostOne
    # 20-29 is Init
    # 30-39 is End
    # 40-139 is RespondedExistence
    # 140-239 is NotRespondedExistence
    # ...
    
    y[_YPos(A, Asq, 'Init', trace[0])] = 1
    y[_YPos(A, Asq, 'End', trace[-1])] = 1

    # first trace loop
    counter = np.zeros(A, dtype=int)
    first_occ = L*np.ones(A, dtype=int) # L means it never occured
    last_occ = -1*np.ones(A, dtype=int) # -1 means it never occured
    alternate_response = np.zeros((A,A), dtype=int) # idea: 0 = all okay; 1 = a occured and b must occur afterwards; -1 = a occured twice
    alternate_precedence = np.zeros((A,A), dtype=int) # idea: 0 = all okay; 1 = a occured and b is allowed to occur from now on; -1 = b occurec twice or without a infront
    chain_response = -1*np.ones(A, dtype=int) # idea: chain response a->b can only be true for one b, maximally. -1 = didn't occur yet. -2 = violated
    chain_precedence = -1*np.ones(A, dtype=int) # idea: chain precedence a->b can only be true for one a, maximally. -1 = didn't occur yet. -2 = violated
    chain_occurrences = np.zeros((A,A), dtype=bool) # idea: check which pairs occur
    for i, a in enumerate(trace):
        counter[a] += 1
        last_occ[a] = i
        if first_occ[a] == L:
            first_occ[a] = i
        for b in range(A):
            if a==b:
                continue
            if alternate_response[a,b] == 0:
                alternate_response[a,b] = 1
            elif alternate_response[a,b] == 1:
                alternate_response[a,b] = -1
            if alternate_response[b,a] == 1:
                alternate_response[b,a] = 0
            if alternate_precedence[a,b] == 0:
                alternate_precedence[a,b] = 1
            if alternate_precedence[b,a] == 0:
                alternate_precedence[b,a] = -1
            elif alternate_precedence[b,a] == 1:
                alternate_precedence[b,a] = 0
        if i != L-1:
            b = trace[i+1]
            if chain_response[a] == -1:
                chain_response[a] = b
            elif chain_response[a] != b:
                chain_response[a] = -2
            if chain_precedence[b] == -1:
                chain_precedence[b] = a
            elif chain_precedence[b] != a:
                chain_precedence[b] = -2
            chain_occurrences[a,b] = True
    chain_response[trace[-1]] = -2 # ending in a results in chain_response(a,.)=False
    chain_precedence[trace[0]] = -2 # starting with b results in chain_precedence(.,b)=False

    # activity loop (squared)
    for a in range(A):
        if counter[a] >= 1:
            y[_YPos(A, Asq, 'AtLeastOne', a)] = 1
        if counter[a] <= 1:
            y[_YPos(A, Asq, 'AtMostOne', a)] = 1
        if chain_response[a] >= 0:
            y[_YPos(A, Asq, 'ChainResponse', a, chain_response[a])] = 1 # ChainResponse 1/2
        if chain_precedence[a] >= 0:
            y[_YPos(A, Asq, 'ChainPrecedence', chain_precedence[a], a)] = 1 # ChainPrecedence 1/2
        for b in range(A):
            if a==b:
                continue
            if (not counter[a]) or counter[b]:
                y[_YPos(A, Asq, 'RespondedExistence', a, b)] = 1
            if (not counter[a]) or (not counter[b]):
                y[_YPos(A, Asq, 'NotRespondedExistence', a, b)] = 1
            if last_occ[a] <= last_occ[b]: # equality fulfills case -1=-1
                y[_YPos(A, Asq, 'Response', a, b)] = 1
            if alternate_response[a,b] == 0:
                y[_YPos(A, Asq, 'AlternateResponse', a, b)] = 1
            if chain_response[a] == -1:
                y[_YPos(A, Asq, 'ChainResponse', a, b)] = 1 # ChainResponse 2/2
            if (not counter[a]) or first_occ[a] >= last_occ[b]:
                y[_YPos(A, Asq, 'NotResponse', a, b)] = 1
            if not chain_occurrences[a,b]:
                y[_YPos(A, Asq, 'NotChainResponse', a, b)] = 1
            if first_occ[a] <= first_occ[b]: # equality fulfills case L=L
                y[_YPos(A, Asq, 'Precedence', a, b)] = 1
            if alternate_precedence[a,b] in (0,1):
                y[_YPos(A, Asq, 'AlternatePrecedence', a, b)] = 1
            if chain_precedence[a] == -1:
                y[_YPos(A, Asq, 'ChainPrecedence', b, a)] = 1 # ChainPrecedence 2/2
            if (not counter[b]) or last_occ[a] <= first_occ[b]:
                y[_YPos(A, Asq, 'NotPrecedence', a, b)] = 1
            if not chain_occurrences[a,b]:
                y[_YPos(A, Asq, 'NotChainPrecedence', a, b)] = 1

    return (case_i, y)