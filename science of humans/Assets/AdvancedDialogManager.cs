using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
/*
 * Le DialogManager permet de centraliser les fonctionnalités liés à l'aspect conversationnel de l'agent. 
 * C'est notamment ici qu'un Dialogue est chargé et est géré. Le DialogManager met à jour l'interface de conversation
 * suivant l'état du Dialogue.
 * Dans cette classe se trouve également des méthodes permettant de maintenir une liste de fichiers audio à jouer pour faire parler l'agent. */
public class AdvancedDialogManager : MonoBehaviour
{
  // Start is called before the first frame update
  public AudioSource audioSource;
  public AudioClip[] audioClips;
  public Animator animator;

  public float volume = 0.5f;
  private int count = 0;
  public int emotion_pos_count = 0;
  public int emotion_neg_count = 0;

  private Dialogue dialog;
  public Transform informationPanel;
  public Transform textPanel;
  public Transform buttonPanel;
  public GameObject ButtonPrefab;

  public bool bonnehumeur = true;


  void Start()
  {

    dialog = Dialogue.readDialogueFile<Multiplications>("Assets/Dialogue/json/multiplications.json", this);
    information_display("");
    dialog.NextQuestion();
    if (bonnehumeur)
    {
      PlaySound(5);
    }
    else
    {
      PlaySound(6);
    }
  }

  // Update is called once per frame
  void Update()
  {

  }
  /*
   * Cette méthode est une méthode de test permettant de jouer le prochain fichier audio dans la liste
   */
  public void PlayNextSound()
  {
    if (count >= audioClips.Length)
      count = 0;
    audioSource.PlayOneShot(audioClips[count], volume);
    count++;
  }

  /*
   * Méthode permettant de jouer le fichier audio en l'appelant par son numéro
   */
  public void PlaySound(int i)
  {
    if (i >= audioClips.Length)
    {
      return;
    }
    else
    {
      audioSource.PlayOneShot(audioClips[i], volume);
    }
  }

  public void information_display(string s)
  {
    Text text = informationPanel.transform.GetComponentInChildren<Text>().GetComponent<Text>();
    text.text = s;
  }

  /* Cette méthode propose des réponses et récupère la réponse.
      Le premier argument correspond à la phrase prononcé par l'agent 
      Le deuxième argument ("proposals") donne la liste des contenus des boutons.
      Le troisième argument ("values") donne la valeur à retourner lorsqu'on clique sur un bouton.
      Les deux listes doivent évidemment faire la même taille... */
  public void agent_dialogue(String s, List<string> proposals, List<System.Object> values)
  {
    if (proposals.Count != values.Count)
    {
      Debug.Log("** Il y a une erreur dans votre code: la liste de proposition et la liste de valeurs ne sont pas de même taille.");
    }

    if (proposals.Count == 0)
    {
      Debug.Log("** Il y a une erreur dans votre code: la liste de proposition est vide.");
    }
    Text text = textPanel.transform.GetComponentInChildren<Text>().GetComponent<Text>();
    text.text = s;

    int i = 0;
    /*On retire tout d'abord tous les boutons de l'interface*/
    foreach (Button child in buttonPanel.transform.GetComponentsInChildren<Button>())
    {
      Destroy(child.gameObject);
    }
    /*Pour chaque valeur, on rajoute un bouton, et on lui associe
        la fonction responseSelected pour quand le bouton est cliqué*/
    for (int j = 0; j < values.Count; j++)
    {
      GameObject button = (GameObject)Instantiate(ButtonPrefab);
      button.GetComponentInChildren<Text>().text = proposals[j];
      System.Object temp = values[j];
      button.GetComponent<Button>().onClick.AddListener(delegate { responseSelected(temp); });
      button.GetComponent<RectTransform>().position = new Vector3(i * 170.0f + 90.0f, 39.0f, 0.0f);
      button.transform.SetParent(buttonPanel);

      Debug.Log(i + ". " + temp);
      i = i + 1;
    }
  }

  /*Quand une réponse est choisie, on appelle la méthode du dialogue qui gère la réponse et ensuite, si le dialogue n'est pas 
      en train de réaliser une action spéciale, on avance dans la question suivante normalement prévu par le JSON. */
  public void responseSelected(System.Object response)
  {
    bool wait = dialog.HandleResponse(response);
    if (!wait)
    {
      information_display("");
      dialog.NextQuestion();
      //Pour l'instant, on ne joue un son que si on suit le JSON, mais on pourrait ajouter des appels de méthodes à cette fonction ailleurs
      //PlayNextSound();
    }
  }
}
