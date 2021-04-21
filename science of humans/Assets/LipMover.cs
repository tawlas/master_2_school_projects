using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LipMover : MonoBehaviour
{

    public AudioSource audioSource;
    public SkinnedMeshRenderer SkinnedMeshRendererTarget = null; ///< As the name implies


    private float referenceTime;
    private int choice;
    private float timeForlerping = 0.175f;

    private int MouthOpenWeight = 0;
    private int UpperLipOutWeight = 0;
    private int LowerLipDown_LeftWeight = 0;
    private int LowerLipDown_RightWeight = 0;
    private int TongueUpWeight = 0;
    private int Smile_RightWeight = 0;
    private int Smile_LeftWeight = 0;
    private int MouthNarrow_LeftWeight = 0;
    private int MouthNarrow_RightWeight = 0;
    private int LowerLipOutWeight = 0;
    private int MouthUpWeight = 0;
    private int LowerLipInWeight = 0;
    private int UpperLipInWeight = 0;
    private int UpperLipUp_LeftWeight = 0;
    private int UpperLipUp_RightWeight = 0;

    private int MouthOpenWeightBack = 0;
    private int UpperLipOutWeightBack = 0;
    private int LowerLipDown_LeftWeightBack = 0;
    private int LowerLipDown_RightWeightBack = 0;
    private int TongueUpWeightBack = 0;
    private int Smile_RightWeightBack = 0;
    private int Smile_LeftWeightBack = 0;
    private int MouthNarrow_LeftWeightBack = 0;
    private int MouthNarrow_RightWeightBack = 0;
    private int LowerLipOutWeightBack = 0;
    private int MouthUpWeightBack = 0;
    private int LowerLipInWeightBack = 0;
    private int UpperLipInWeightBack = 0;
    private int UpperLipUp_LeftWeightBack = 0;
    private int UpperLipUp_RightWeightBack = 0;

    void Start()
    {
        referenceTime = Time.time;
        audioSource = GetComponent<AudioSource>();
        if (SkinnedMeshRendererTarget == null)
            SkinnedMeshRendererTarget = gameObject.GetComponent<SkinnedMeshRenderer>();


    }

    void Update()
    {
        float now = Time.time;
        float lerp = (now - referenceTime) / timeForlerping;
        if (SkinnedMeshRendererTarget != null)
        {

            if (!audioSource.isPlaying)
            {
                UpdateBackWeight();
                setVisemeNeutral();
            }
            else
            {

                if (now - referenceTime > timeForlerping)
                {
                    UpdateBackWeight();
                    choice = Random.Range(0, 11);
                    referenceTime = Time.time;
                    setRandomViseme(choice);
                }
            }
            lerpViseme(lerp);
        }
    }

    public void UpdateBackWeight()
    {
        MouthOpenWeightBack = MouthOpenWeight;
        UpperLipOutWeightBack = UpperLipOutWeight;
        LowerLipDown_LeftWeightBack = LowerLipDown_LeftWeight;
        LowerLipDown_RightWeightBack = LowerLipDown_RightWeight;
        TongueUpWeightBack = TongueUpWeight;
        Smile_RightWeightBack = Smile_RightWeight;
        Smile_LeftWeightBack = Smile_LeftWeight;
        MouthNarrow_LeftWeightBack = MouthNarrow_LeftWeight;
        MouthNarrow_RightWeightBack = MouthNarrow_RightWeight;
        LowerLipOutWeightBack = LowerLipOutWeight;
        MouthUpWeightBack = MouthUpWeight;
        LowerLipInWeightBack = LowerLipInWeight;
        UpperLipInWeightBack = UpperLipInWeight;
        UpperLipUp_LeftWeightBack = UpperLipUp_LeftWeight;
        UpperLipUp_RightWeightBack = UpperLipUp_RightWeight;
    }

    /*!
       * @brief A function for getting blendshape index by name.
       * @return int
       */
    public int getBlendShapeIndex(SkinnedMeshRenderer smr, string bsName)
    {
        Mesh m = smr.sharedMesh;

        for (int i = 0; i < m.blendShapeCount; i++)
        {
            string name = m.GetBlendShapeName(i);
            if (bsName.Equals(m.GetBlendShapeName(i)) == true)
                return i;
        }

        return 0;
    }

    public void setRandomViseme(int choice)
    {

        switch (choice)
        {
            case 0: setViseme_R_ER(); break;
            case 1: setViseme_AA_AO_OW(); break;
            case 2: setViseme_m_b_p_x(); break;
            case 3: setViseme_N_NG_CH_TH_ZH_DH_j_s(); break;
            case 4: setViseme_y_EH_IY(); break;
            case 5: setViseme_L_EL(); break;
            case 6: setViseme_w(); break;
            case 7: setViseme_IH_AH_AE(); break;
            case 8: setViseme_U_UW(); break;
            case 9: setViseme_AW(); break;
            case 10: setViseme_fv(); break;
            default: setViseme_AA_AO_OW(); break;

        }
    }

    public void setVisemeNeutral()
    {
        MouthOpenWeight = 0;
        UpperLipOutWeight = 0;
        LowerLipDown_LeftWeight = 0;
        LowerLipDown_RightWeight = 0;
        TongueUpWeight = 0;
        Smile_RightWeight = 0;
        Smile_LeftWeight = 0;
        MouthNarrow_LeftWeight = 0;
        MouthNarrow_RightWeight = 0;
        LowerLipOutWeight = 0;
        MouthUpWeight = 0;
        LowerLipInWeight = 0;
        UpperLipInWeight = 0;
        UpperLipUp_LeftWeight = 0;
        UpperLipUp_RightWeight = 0;
    }

    public void lerpViseme(float lerp)
    {
        Mesh m = SkinnedMeshRendererTarget.sharedMesh;
        int i = getBlendShapeIndex(SkinnedMeshRendererTarget, "MouthOpen");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(MouthOpenWeightBack, MouthOpenWeight, lerp));

        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "MouthUp");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(MouthUpWeightBack, MouthUpWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "LowerLipOut");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(LowerLipOutWeightBack, LowerLipOutWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "MouthNarrow_Left");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(MouthNarrow_LeftWeightBack, MouthNarrow_LeftWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "MouthNarrow_Right");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(MouthNarrow_RightWeightBack, MouthNarrow_RightWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "UpperLipOut");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(UpperLipOutWeightBack, UpperLipOutWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "LowerLipIn");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(LowerLipInWeightBack, LowerLipInWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "UpperLipIn");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(UpperLipInWeightBack, UpperLipInWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "LowerLipDown_Left");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(LowerLipDown_LeftWeightBack, LowerLipDown_LeftWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "LowerLipDown_Right");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(LowerLipDown_RightWeightBack, LowerLipDown_RightWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "UpperLipUp_Left");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(UpperLipUp_LeftWeightBack, UpperLipUp_LeftWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "UpperLipUp_Right");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(UpperLipUp_RightWeightBack, UpperLipUp_RightWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "Smile_Right");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(Smile_RightWeightBack, Smile_RightWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "Smile_Left");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(Smile_LeftWeightBack, Smile_LeftWeight, lerp));
        i = getBlendShapeIndex(SkinnedMeshRendererTarget, "TongueUp");
        SkinnedMeshRendererTarget.SetBlendShapeWeight(i, (int)Mathf.Lerp(TongueUpWeightBack, TongueUpWeight, lerp));




    }


    public void setViseme_AA_AO_OW()
    {
        setVisemeNeutral();
        LowerLipOutWeight = 75;
        MouthNarrow_LeftWeight = 75;
        MouthNarrow_RightWeight = 75;
        //MouthOpenWeight = 30;
        UpperLipOutWeight = 75;


    }
    public void setViseme_R_ER()
    {
        setVisemeNeutral();
        LowerLipOutWeight = 75;
        MouthNarrow_LeftWeight = 75;
        MouthNarrow_RightWeight = 75;
        //MouthOpenWeight = 30;
        UpperLipOutWeight = 75;

    }
    public void setViseme_m_b_p_x()
    {
        setVisemeNeutral();
        LowerLipInWeight = 75;
        UpperLipInWeight = 75;

    }
    public void setViseme_N_NG_CH_TH_ZH_DH_j_s()
    {
        setVisemeNeutral();
        //MouthOpenWeight = 30;

    }
    public void setViseme_y_EH_IY()
    {
        setVisemeNeutral();
        LowerLipDown_LeftWeight = 30;
        LowerLipDown_RightWeight = 30;
        //MouthOpenWeight = 15;
        UpperLipUp_LeftWeight = 30;
        UpperLipUp_RightWeight = 30;
        Smile_RightWeight = 30;
        Smile_LeftWeight = 30;

    }
    public void setViseme_L_EL()
    {
        setVisemeNeutral();
        //MouthOpenWeight = 60;
        TongueUpWeight = 75;

    }
    public void setViseme_w()
    {
        setVisemeNeutral();
        //MouthOpenWeight = 25;
        MouthNarrow_LeftWeight = 75;
        MouthNarrow_RightWeight = 75;

    }
    public void setViseme_IH_AH_AE()
    {
        setVisemeNeutral();
        //MouthOpenWeight = 50;
        Smile_RightWeight = 50;
        Smile_LeftWeight = 50;

    }
    public void setViseme_U_UW()
    {
        setVisemeNeutral();
        MouthNarrow_LeftWeight = 75;
        MouthNarrow_RightWeight = 75;
        LowerLipOutWeight = 75;
        UpperLipInWeight = 75;

    }
    public void setViseme_AW()
    {
        setVisemeNeutral();
        //MouthOpenWeight = 30;
        MouthUpWeight = 30;
        UpperLipUp_LeftWeight = 30;
        UpperLipUp_RightWeight = 75;

    }
    public void setViseme_fv()
    {
        setVisemeNeutral();
        LowerLipInWeight = 75;
        UpperLipInWeight = 75;
        UpperLipUp_LeftWeight = 50;
        UpperLipUp_RightWeight = 50;

    }



}