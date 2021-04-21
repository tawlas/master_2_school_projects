using System.Collections;
using System.Collections.Generic;
using UnityEngine;


class Multiplications : Dialogue
{
    /* la constante MAX_TABLE vaut 10 : nous apprenons les tables de 1 à 10. */
    const int MAX_TABLE = 10;
    /* la variable vus permet de savoir combien de fois chaque table a été vue. */
    private int[] vus = new int[MAX_TABLE + 1];

    /* une variable aléatoire partagée */
    private static Random rand = new Random();

    /*la table à afficher*/
    private int table = 0;

    /*la ligne de texte à dire*/
    protected string ligne = "";

    /* le constructeur par défaut de la classe (c'est le seul utilisable car la classe
       est parsée par JSON) permet d'initialiser la variable vus. */
    public Multiplications()
    {
        for (int i = 0; i < MAX_TABLE + 1; i++)
            vus[i] = 0;
    }

    /* Cette méthode permet d'afficher la table de multiplication de "n" (n entre 1 et MAX_TABLE). */
    private void afficher_table(int n)
    {
        if (n < 1 || n > MAX_TABLE)
            return;
        vus[n]++;
        ligne = "*** Table de " + n + " ***\n";
        for (int i = 0; i <= 10; i++)
            ligne += i + "\tx" + n + "\t=" + i * n + "\n";
        DialogManager.information_display(ligne);
        ligne = "Dites-moi lorsque vous avez fini d'apprendre cette table.";
        List<string> rep_s = new List<string>();
        List<System.Object> val_s = new List<System.Object>();
        rep_s.Add("Ok!");
        //On insère dans la valeur de réponse, l'état du Dialogue où l'on veut reprendre après cette action spéciale
        val_s.Add("etat|debut");
        DialogManager.agent_dialogue(ligne,rep_s, val_s);

    }



    /* une méthode pour mélanger */
    private static void shuffle(List<int> list)
    {
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = Random.Range(0, n);
            int value = list[k];
            list[k] = list[n];
            list[n] = value;

        }
    }

    /* Interroge sur une table et affiche une évaluation (neutre) */
    void interroge_table(int n)
    {
        /* affichage de la question */
        int i = Random.Range(1, MAX_TABLE + 1);
        //DialogManager.agent_display("** Question : combien font " + i + " x " + n + " ?");
        /* réponses possibles */
        List<int> rep_v = new List<int>();
        rep_v.Add(i * n);
        rep_v.Add(i * n + Random.Range(1, 20));
        rep_v.Add(i * n - Random.Range(1, i * n > 20 ? i * n - 20 : i * n));
        int h;
        do { h = Random.Range(1, MAX_TABLE + 1); } while (h == i || rep_v.Contains(h * n));
        rep_v.Add(h * n);
        do { h = Random.Range(1, MAX_TABLE + 1); } while (h == n || rep_v.Contains(h * i));
        rep_v.Add(h * i);
        shuffle(rep_v);
        /* affichage */
        List<string> rep_s = new List<string>();
        List<System.Object> val_s = new List<System.Object>();
        //ATTENTION : On insère dans la valeur de la réponse
        //l'état du script JSON où l'on veut reprendre le dialogue ensuite
        //si la réponse est la bonne, on insère que l'on veut reprendre le dialogue à l'état : bonnereponse
        //si la réponse est mauvaise, on insère que l'on veut reprendre le dialogue à l'état : mauvaisereponse
        foreach (int x in rep_v)
        {
            rep_s.Add(x.ToString());
            if (x == i * n)
            {
                val_s.Add("etat|bonnereponse");
            }
            else
            {
                val_s.Add("etat|mauvaisereponse");
            }
        }
        rep_s.Add("Je ne sais pas, je n'ai pas vu cette table.");
        val_s.Add("etat|apprendre");
        DialogManager.agent_dialogue("** Question : combien font " + i + " x " + n + " ?",rep_s, val_s);

    }


    /* Interroge sur n'importe quelle table */
    void interroger()
    {
        interroge_table(Random.Range(1, MAX_TABLE + 1));
    }

    /*
     * La méthode action est appellée après chaque réception de réponse.
     * On y vérifie les nouvelles valeurs du Dialogue et on effectue éventuellement quelques actions liées à ces valeurs.
     * Si l'on souhaite entrer dans un état temporaire du dialogue (non prévu par le scrip JSON) et effectuer des actions spéciales,
     * il est nécessaire de renvoyer TRUE à la fin de cette méthode afin que la méthode NextQuestion ne soit pas appelée.*/
    public override bool action(System.Object param)
    {
         /* on quitte */
        if (hasValue("etat", "kill"))
        {
            Debug.Log("Number of positive expressions played:");
            Debug.Log(DialogManager.emotion_pos_count.ToString());
            Debug.Log("Number of negative expressions played:");
            Debug.Log(DialogManager.emotion_neg_count.ToString());
            // Jouons le son pour dire au revoir
            DialogManager.PlaySound(7);
            Application.Quit();
        }
        //par défaut, on renvoie faux, le Dialogue suit son cours normal
        bool wait = false;


        //ici on vérifie si dans le paramètre de la réponse ne se trouve pas un état de retour.
        //Si c'est le cas, on le prend en compte
        if (param is string && ((string)param).Split('|')[0] == "etat")
        {
            setValue(((string)param).Split('|')[0], ((string)param).Split('|')[1]);
        }
        else
        {
           
            if (hasValue("etat", "tableran"))
            {

                table = Random.Range(1, MAX_TABLE + 1);
                bool ok = false;
                for (int i = 1; i < MAX_TABLE + 1; i++)
                    if (vus[i] == 0)
                    {
                        ok = true;
                        break;
                    }
                if (!ok)
                    table = Random.Range(1, MAX_TABLE + 1);


                /* ... puis on affiche la table */
                
                
                afficher_table(table);
                wait = true;

            }
            //Pour chacune des actions spéciales à effectuer, il est nécessaire de renvoyer true.
            if (hasValue("etat", "table"))
            {
               
                afficher_table(int.Parse((string)getValue("value")));
                wait = true;
            }
            /* interrogation puis on passe à la question "recommencer" */
            else if (hasValue("etat", "reciter"))
            {
                // Jouer le son de la question
                DialogManager.PlaySound(0);
                interroger();
                wait = true;

            }

        }

        // Réaction émotive à la réponse donnée par l'utilisateur, selon l'humeur de l'agent.
        if (DialogManager.bonnehumeur)
        {
            if (hasValue("etat", "bonnereponse"))
            {
                DialogManager.emotion_pos_count++;
                DialogManager.animator.SetTrigger("Joy");
                DialogManager.PlaySound(1);
            }
            else if (hasValue("etat", "mauvaisereponse"))
            {
                DialogManager.emotion_neg_count++;
                DialogManager.animator.SetTrigger("Sad");
                DialogManager.PlaySound(2);
            }
        }
        else
        {
            if (hasValue("etat", "bonnereponse"))
            {
                DialogManager.emotion_pos_count++;
                DialogManager.animator.SetTrigger("Anger");
                DialogManager.PlaySound(3);
            }
            else if (hasValue("etat", "mauvaisereponse"))
            {
                DialogManager.emotion_neg_count++;
                DialogManager.animator.SetTrigger("Mocking");
                DialogManager.PlaySound(4);
            }
        }
        

        /*
         de manière aléatoire, on glisse le dialogue sur l'humeur 
        if (hasValue("raison", "ignore") && Random.Range(0.0f, 1.0f) < 0.25)
        {
            setValue("humeur", "inconnue");
            setValue("raison", "inconnue");
        }
         et quand la question a été posée, on la remet à ignore 
        else if (!hasValue("humeur", "ignore") && !hasValue("raison", "inconnue"))
        {
            // en réalité, avant de les supprimer, il faudra faire quelque chose de ces valeurs...
            setValue("humeur", "ignore");
            setValue("raison", "ignore");
        } */
        return wait;

    }
    public override bool done()
    {
        return hasValue("etat", "kill");
    }

}