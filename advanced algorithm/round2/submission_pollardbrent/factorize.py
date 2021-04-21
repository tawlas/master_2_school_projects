import sys, os

def factorize(numbers):
    """
        A Faire:         
        - Ecrire une fonction qui prend en paramètre une liste de nombres et qui retourne leurs decompositions en facteurs premiers
        - cette fonction doit retourner un dictionnaire Python où :
            -- la clé est un nombre n parmi la liste de nombres en entrée
            -- la valeur est la liste des facteurs premiers de n (clé). Leur produit correpond à n (clé).  
            
        - Attention : 
            -- 1 n'est pas un nombre premier
            -- un facteur premier doit être répété autant de fois que nécessaire. Chaque nombre est égale au produit de ses facteurs premiers. 
            -- une solution partielle est rejetée lors de la soumission. Tous les nombres en entrée doivent être traités. 
            -- Ne changez pas le nom de cette fonction, vous pouvez ajouter d'autres fonctions appelées depuis celle-ci.
            -- Ne laissez pas trainer du code hors fonctions car ce module sera importé et du coup un tel code sera exécuté et cela vous pénalisera en temps.
    """
    def primesbelow(N):
        correction = N % 6 > 1
        N = {0:N, 1:N-1, 2:N+4, 3:N+3, 4:N+2, 5:N+1}[N%6]
        sieve = [True] * (N // 3)
        sieve[0] = False
        for i in range(int(N ** .5) // 3 + 1):
            if sieve[i]:
                k = (3 * i + 1) | 1
                sieve[k*k // 3::2*k] = [False] * ((N//6 - (k*k)//6 - 1)//k + 1)
                sieve[(k*k + 4*k - 2*k*(i%2)) // 3::2*k] = [False] * ((N // 6 - (k*k + 4*k - 2*k*(i%2))//6 - 1) // k + 1)
        return [2, 3] + [(3 * i + 1) | 1 for i in range(1, N//3 - correction) if sieve[i]]

    smallprimeset = set(primesbelow(100000))
    _smallprimeset = 100000
    def isprime(n, precision=7):
        if n < 1:
            raise ValueError("n doit etre plus grand que 0")
        elif n <= 3:
            return n >= 2
        elif n % 2 == 0:
            return False
        elif n < _smallprimeset:
            return n in smallprimeset


        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1

        for repeat in range(precision):
            a = 2
            x = pow(a, d, n)

            if x == 1 or x == n - 1: continue

            for r in range(s - 1):
                x = pow(x, 2, n)
                if x == 1: return False
                if x == n - 1: break
            else: return False

        return True

    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3

        y, c, m = 1, 1, 1
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for i in range(r):
                y = (pow(y, 2, n) + c) % n

            k = 0
            while k < r and g==1:
                ys = y
                for i in range(min(m, r-k)):
                    y = (pow(y, 2, n) + c) % n
                    q = q * abs(x-y) % n
                g = gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (pow(ys, 2, n) + c) % n
                g = gcd(abs(x - ys), n)
                if g > 1:
                    break

        return g

    smallprimes = primesbelow(1000) 
    def primefactors(n, sort=True):
        factors = []

        for checker in smallprimes:
            while n % checker == 0:
                factors.append(checker)
                n //= checker
            if checker > n: break

        if n < 2: return factors

        while n > 1:
            if isprime(n):
                factors.append(n)
                break
            factor = pollard_brent(n) 
            factors.extend(primefactors(factor))
            n //= factor

        if sort: factors.sort()

        return factors


    totients = {}
    def totient(n):
        if n == 0: return 1

        try: return totients[n]
        except KeyError: pass

        tot = 1
        for p, exp in factorization(n).items():
            tot *= (p - 1)  *  p ** (exp - 1)

        totients[n] = tot
        return tot

    def gcd(a, b):
        if a == b: return a
        while b > 0: a, b = b, a % b
        return a

    def lcm(a, b):
        return abs((a // gcd(a, b)) * b)
        
    result = {}
    for n in numbers:
        factors = primefactors(n)
        result[n] = factors
    return result

#########################################
#### Ne pas modifier le code suivant ####
#########################################
if __name__=="__main__":
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    
    # un repertoire des fichiers en entree doit être passé en parametre 1
    if not os.path.isdir(input_dir):
	    print(input_dir, "doesn't exist")
	    exit()

    # un repertoire pour enregistrer les résultats doit être passé en parametre 2
    if not os.path.isdir(output_dir):
	    print(output_dir, "doesn't exist")
	    exit()       

     # Pour chacun des fichiers en entrée 
    for data_filename in sorted(os.listdir(input_dir)):

        
        # importer la liste des nombres
        data_file = open(os.path.join(input_dir, data_filename), "r")
        numbers = [int(line) for line in data_file.readlines()]   
        # decomposition en facteurs premiers
        D = factorize(numbers)

        # fichier des reponses depose dans le output_dir
        output_filename = 'answer_{}'.format(data_filename)             
        output_file = open(os.path.join(output_dir, output_filename), 'w')
        
        # ecriture des resultats
        for (n, primes) in D.items():
            output_file.write('{} {}\n'.format(n, primes))
        
        output_file.close()
        
        

    
