using System;                           // pour les entrées-sorties (Console)
using System.Collections.Generic;       // pour les List<T>
using System.IO;                        // pour la lecture et l'écriture de fichiers
using Newtonsoft.Json;                  // pour la manipulation de fichiers JSON
using UnityEngine;


/*  La classe abstraite dialogue est le support pour définir des dialogues.
 *  Celle-ci est dépendante de la classe AdvancedDialogManager pour la gestion de l'interaction du dialogue

    On définit un ensemble de variables.
    On définit un ensemble de questions déclenchées par les valeurs des variables
    Pour chaque question, on définit un ensemble de réponses.
    Pour chaque réponse, on définit les nouvelles valeurs des variables.
    Le dialogue fonctionne de la manière suivante :
    - sélection de la question (la première qui satisfait les conditions)
    - attente de la réponse (via les boutons)
    - modification des variables
    - appel de la méthode abstraite "action"
    - recommencer tant que la méthode abstraite "done" ne renvoie pas true

    
    Dans l'écriture des méthode action et done, on peut utiliser les valeurs
    des variables qui sont manipulées par le dialogue. Pour cela, on dispose
    des méthodes getValue, hasValue et setValue.

*/
public abstract class Dialogue
{
    private AdvancedDialogManager dialogManager;
    /************************************************/
    /* Structure C# du dialogue, alimentée via JSON */
    /************************************************/

    /* La classe pour les variables : un nom associé à une valeur */
    public class Variable
    {
        public string name { get; set; }
        public object val { get; set; }
    }

    /* La classe pour les réponses */
    public class Response
    {
        public string phrase { get; set; }
        public List<Variable> vars { get; set; }
    }

    /* La classe pour les questions */
    public class Question
    {
        public List<Variable> conditions { get; set; }
        public string phrase { get; set; }
        public List<Response> responses { get; set; }
    }

    /* Maintenant, un Dialogue est un ensemble de variables et de questions */
    public List<Variable> vars { get; set; }
    public List<Question> questions { get; set; }
    /* Le dialogue devra avoir une référence vers le DialogManager afin de lui renvoyer les nouveaux états du Dialogue */
    public AdvancedDialogManager DialogManager { get => dialogManager; set => dialogManager = value; }
    

    /**********************************/
    /* Méthodes de la classe Dialogue */
    /**********************************/

    /* Il faudra instancier la méthode abstraite "action"... 
     Le booleen en retour devrait être utilisé pour indiquer 
     au DialogManager que le Dialogue est en train de réaliser une action spéciale 
     et qu'il ne faut pas passer tout de suite à la question suivante du script JSON*/
    public abstract bool action(System.Object param);
    /* ... et la méthode abstraite "done"... */
    public abstract bool done();

    /* Une méthode bien pratique pour récupérer la valeur d'une variable du dialogue */
    public object getValue(string name)
    {
        foreach (Variable v in vars)
            if (v.name.Equals(name))
                return v.val;
         Debug.Log("** Il y a une erreur dans votre dialogue ou dans votre code:");
         Debug.Log("** La variable "+name+" ne figure pas dans le dialogue");
        Environment.Exit(0);
        return null;
    }

    /* Une autre méthode bien pratique pour tester la valeur d'une variable du dialogue
        ATTENTION : on retranstype tout en "string" pour comparer les valeurs (ils étaient de la classe Object avant) !
    */
    public bool hasValue(string name, object val)
    {
        string current_value = getValue(name).ToString();
        string target_value = val.ToString();
        return current_value.Equals(target_value);
    }

    /* Une dernière méthode bien pratique pour modifier la valeur d'une variable du dialogue */
    public void setValue(string name, object val)
    {
        foreach (Variable v in vars)
            if (v.name.Equals(name))
            {
                v.val = val;
                return;
            }
         Debug.Log("** Il y a une erreur dans votre dialogue ou dans votre code:");
         Debug.Log("** La variable "+ name + " ne figure pas dans le dialogue");
        Environment.Exit(0);
    }

    /* La méthode pour lire un dialogue
        ATTENTION : T doit être une sous-classe de Dialogue
    */
    public static T readDialogueFile<T>(string filename, AdvancedDialogManager menu) where T : Dialogue
    {
        
        string sdial = File.ReadAllText(filename);
        T toRet = JsonConvert.DeserializeObject<T>(sdial);
        toRet.DialogManager = menu;
        return toRet;
    }

    /*
     * La méthode NextQuestion prend la première question dans le JSON qui 
     * satisfait les conditions d'état des variables.
     * (On peut dans le code, venir altérer l'état de ces variables pour venir
     * changer dynamiquement le déroulé originellement prévu dans le fichier JSON)
     */ 
    public Question current_q = null;
    public void NextQuestion()
    {
        /* 0. Test de fin */
        if(!done())
        {
            /* 1. Sélection de la question */
            foreach (Question q in questions)
            {
                /* vérification des conditions */
                bool ok = true;
                foreach (Variable v in q.conditions)
                {
                    if (!hasValue(v.name, v.val))
                    {
                        ok = false;
                        break;
                    }
                }
                /* si toutes les conditions sont satisfaites, on part sur cette question */
                if (ok)
                {
                    current_q = q;
                    break;
                }
            }
            /* car d'erreur : aucune question possible */
            if (current_q == null)
            {
                Debug.Log("** Il y a une erreur dans votre dialogue:");
                Debug.Log("** Aucune question ne satisfait les conditions courantes:");
                foreach (Variable vv in vars)
                    Debug.Log(vv.name+" = "+ vv.val);
                
            }
            /* 2. Récupération des réponses possibles */
            List<string> props = new List<string>();
            List<System.Object> vals = new List<System.Object>();
            int i = 0;
            foreach (Response rep in current_q.responses)
            {
                props.Add(rep.phrase);
                vals.Add(i);
                i = i + 1;
            }
            /* 3. Création de la ligne de texte dans l'interface correspond à la question et création des boutons
             * correspondants aux réponses avec leurs valeurs de retours*/
            dialogManager.agent_dialogue(current_q.phrase,props, vals);
           
        }
    }
    
    /* Cette méthode gère la récupération de la réponse */
    public bool HandleResponse(System.Object param)
    {
        /* 4. Si le Dialogue était en train de réaliser une action spéciale (sans faire avancer le script JSON)
            alors le paramètre de retour ne devrait pas être un entier (voir l'exemple de Multiplications
            qui ne renvoit pas un entier lorsque une des actions prend la main sur le dialogue prévu par le JSON). 
            
            Sinon on met à jour normalement les valeurs provenant du Script JSON.
         */
        if (param is int)
        {
            Response current_r = current_q.responses[(int)param];
            foreach (Variable v in current_r.vars)
                setValue(v.name, v.val);
        }
        /* 5. appel de la méthode abstraite "action" */
        return action(param);

    }

}


/* La classe Dialogue0 est un exemple minimaliste de code,
    utilisé pour tester les fonctionnalités du système de dialogue.
*/
/* ce code correspond au tout premier dialogue */
class Dialogue0 : Dialogue
{
    public override bool action(System.Object param)
    {
        if (!hasValue("etat", "fin"))
        {
            setValue("etat", "debut");
        }
        return false;
    }
    public override bool done()
    {
        return hasValue("etat", "fin");
    }
    
}
