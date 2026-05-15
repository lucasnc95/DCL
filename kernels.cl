//*****************
//Orientacao Malha.
//*****************
//
//	 (Z)
//    |
//	  |
//	  |____(X)
//    /
//   /
//  /
//(Y)
//
//*****************

//Tipos de celulas.
#define MALHA_TOTAL_CELULAS	8
#define CELULA_A		0
#define CELULA_MR		1
#define CELULA_MA		2
#define CELULA_N		3
#define CELULA_CH		4
#define CELULA_ND		5
#define CELULA_G		6
#define CELULA_CA		7

//Estrutura de celulas.
#define CELULAS_SIZEOF			64
#define CELULAS_SINGLE_CELL_SIZEOF	8

#define CELULAS_A_OFFSET		0
#define CELULAS_MR_OFFSET		8
#define CELULAS_MA_OFFSET		16
#define CELULAS_N_OFFSET		24
#define CELULAS_CH_OFFSET		32
#define CELULAS_ND_OFFSET		40
#define CELULAS_G_OFFSET		48
#define CELULAS_CA_OFFSET		56

#define CELULAS_POSICAO_OR_OFFSET	0
#define CELULAS_POSICAO_XP_OFFSET	1
#define CELULAS_POSICAO_XM_OFFSET	2
#define CELULAS_POSICAO_YP_OFFSET	3
#define CELULAS_POSICAO_YM_OFFSET	4
#define CELULAS_POSICAO_ZP_OFFSET	5
#define CELULAS_POSICAO_ZM_OFFSET	6
#define CELULAS_NOVO_VALOR_OFFSET	7

//Informacoes de acesso à estrutura "parametrosMalhaGPU".
#define OFFSET_COMPUTACAO		0
#define LENGTH_COMPUTACAO		1
#define MALHA_DIMENSAO_X		2
#define MALHA_DIMENSAO_Y		3
#define MALHA_DIMENSAO_Z		4
#define MALHA_DIMENSAO_POSICAO_Z	5
#define MALHA_DIMENSAO_POSICAO_Y	6
#define MALHA_DIMENSAO_POSICAO_X	7
#define MALHA_DIMENSAO_CELULAS		8
#define NUMERO_PARAMETROS_MALHA		9

#define days 1

__constant float deltaT = 1 / ((float)days * 1000000);

__constant float keqch = 1.0;
__constant float Pmax = 11.4;
__constant float Pmin = 0.0001;
__constant float NMaxTissue = 8;
__constant float MrPMax = 0.1;
__constant float MrPMin = 0.01;
__constant float MrMaxTissue = 6;
__constant float maActivationRate = 0.1;

__constant float m_a = 0;
__constant float l_n_a = 0.55;
__constant float l_ma_a = 0.8;
__constant float d_a = 2000 * 0.001;

__constant float m_mr = 0.033;
__constant float d_mr = 4320 * 0.001;
__constant float x_mr = 3600 * 0.001;

__constant float m_ma = 0.07;
__constant float d_ma = 3000 * 0.001;
__constant float x_ma = 4320 * 0.001;

__constant float m_n = 3.43;
__constant float d_n = 12096 * 0.001;
__constant float x_n = 14400 * 0.001;

__constant float m_ch = 7;
__constant float b_ch_n = 1;
__constant float b_ch_ma = 0.8;
__constant float d_ch = 9216 * 0.001;
__constant float chInf = 3.6;

__constant float l_nd_ma = 2.6;
__constant float d_nd = 0.144 * 0.001;

__constant float m_g = 5;
__constant float b_g_n = 0.6;
__constant float d_g = 9216 * 0.001;

__constant float m_ca = 4;
__constant float b_ca_mr = 1.5;
__constant float b_ca_ma = 1.5;
__constant float d_ca = 9216 * 0.001;

__constant float caInf = 3.6;
__constant float gInf = 3.1;

__constant float deltaX = 0.1;
__constant float deltaY = 0.1;
__constant float deltaZ = 0.1;


// Mantém a mesma condição de borda usada no código original.
// Interior: diferença central de segunda ordem.
// Borda inferior: usa vizinho positivo.
// Borda superior: usa vizinho negativo.
float SegundaDerivada(
    int posicao,
    int dimensao,
    float valorCentral,
    float valorPositivo,
    float valorNegativo,
    float delta)
{
    if (posicao > 0 && posicao < dimensao - 1)
    {
        return (valorPositivo - 2.0f * valorCentral + valorNegativo) / (delta * delta);
    }

    if (posicao == 0)
    {
        return (valorPositivo - valorCentral) / (delta * delta);
    }

    return (valorNegativo - valorCentral) / (delta * delta);
}


// CORRIGIDO:
// X usa XP/XM.
// Y usa YP/YM.
// Z usa ZP/ZM.
float Laplaciano(
    int celulaOffset,
    float *celulas,
    int xPosicaoGlobal,
    int yPosicaoGlobal,
    int zPosicaoGlobal,
    __constant int *parametrosMalhaGPU)
{
    float centro = celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET];

    float lapX = SegundaDerivada(
        xPosicaoGlobal,
        parametrosMalhaGPU[MALHA_DIMENSAO_X],
        centro,
        celulas[celulaOffset + CELULAS_POSICAO_XP_OFFSET],
        celulas[celulaOffset + CELULAS_POSICAO_XM_OFFSET],
        deltaX);

    float lapY = SegundaDerivada(
        yPosicaoGlobal,
        parametrosMalhaGPU[MALHA_DIMENSAO_Y],
        centro,
        celulas[celulaOffset + CELULAS_POSICAO_YP_OFFSET],
        celulas[celulaOffset + CELULAS_POSICAO_YM_OFFSET],
        deltaY);

    float lapZ = SegundaDerivada(
        zPosicaoGlobal,
        parametrosMalhaGPU[MALHA_DIMENSAO_Z],
        centro,
        celulas[celulaOffset + CELULAS_POSICAO_ZP_OFFSET],
        celulas[celulaOffset + CELULAS_POSICAO_ZM_OFFSET],
        deltaZ);

    return lapX + lapY + lapZ;
}


float Quimiotaxia(
    int celulaOffset,
    float *celulas,
    int xPosicaoGlobal,
    int yPosicaoGlobal,
    int zPosicaoGlobal,
    __constant int *parametrosMalhaGPU)
{
    return
        (
            (
                (
                    (xPosicaoGlobal > 0)
                    ?
                    (
                        (
                            (celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                             celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_XM_OFFSET]) > 0
                        )
                        ?
                        (
                            -(
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_XM_OFFSET]
                             )
                             * celulas[celulaOffset + CELULAS_POSICAO_XM_OFFSET]
                             / deltaX
                        )
                        :
                        (
                            -(
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_XM_OFFSET]
                             )
                             * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]
                             / deltaX
                        )
                    )
                    :
                    0
                )
                +
                (
                    (xPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_X] - 1)
                    ?
                    (
                        (
                            (celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_XP_OFFSET] -
                             celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]) > 0
                        )
                        ?
                        (
                            (
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_XP_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                            )
                            * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]
                            / deltaX
                        )
                        :
                        (
                            (
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_XP_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                            )
                            * celulas[celulaOffset + CELULAS_POSICAO_XP_OFFSET]
                            / deltaX
                        )
                    )
                    :
                    0
                )
            )
            / deltaX
        )
        +
        (
            (
                (
                    (yPosicaoGlobal > 0)
                    ?
                    (
                        (
                            (celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                             celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_YM_OFFSET]) > 0
                        )
                        ?
                        (
                            -(
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_YM_OFFSET]
                             )
                             * celulas[celulaOffset + CELULAS_POSICAO_YM_OFFSET]
                             / deltaY
                        )
                        :
                        (
                            -(
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_YM_OFFSET]
                             )
                             * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]
                             / deltaY
                        )
                    )
                    :
                    0
                )
                +
                (
                    (yPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_Y] - 1)
                    ?
                    (
                        (
                            (celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_YP_OFFSET] -
                             celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]) > 0
                        )
                        ?
                        (
                            (
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_YP_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                            )
                            * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]
                            / deltaY
                        )
                        :
                        (
                            (
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_YP_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                            )
                            * celulas[celulaOffset + CELULAS_POSICAO_YP_OFFSET]
                            / deltaY
                        )
                    )
                    :
                    0
                )
            )
            / deltaY
        )
        +
        (
            (
                (
                    (zPosicaoGlobal > 0)
                    ?
                    (
                        (
                            (celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                             celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_ZM_OFFSET]) > 0
                        )
                        ?
                        (
                            -(
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_ZM_OFFSET]
                             )
                             * celulas[celulaOffset + CELULAS_POSICAO_ZM_OFFSET]
                             / deltaZ
                        )
                        :
                        (
                            -(
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_ZM_OFFSET]
                             )
                             * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]
                             / deltaZ
                        )
                    )
                    :
                    0
                )
                +
                (
                    (zPosicaoGlobal < parametrosMalhaGPU[MALHA_DIMENSAO_Z] - 1)
                    ?
                    (
                        (
                            (celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_ZP_OFFSET] -
                             celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]) > 0
                        )
                        ?
                        (
                            (
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_ZP_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                            )
                            * celulas[celulaOffset + CELULAS_POSICAO_OR_OFFSET]
                            / deltaZ
                        )
                        :
                        (
                            (
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_ZP_OFFSET] -
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                            )
                            * celulas[celulaOffset + CELULAS_POSICAO_ZP_OFFSET]
                            / deltaZ
                        )
                    )
                    :
                    0
                )
            )
            / deltaZ
        );
}


void CalcularPontos(
    float *celulas,
    int xPosicaoGlobal,
    int yPosicaoGlobal,
    int zPosicaoGlobal,
    __constant int *parametrosMalhaGPU)
{
    // Antigenos
    float maActivation =
        maActivationRate *
        celulas[CELULAS_MR_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
        celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET];

    celulas[CELULAS_A_OFFSET + CELULAS_NOVO_VALOR_OFFSET] =
        max(
            0.0f,
            (
                (-1 * m_a * celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET])
                - maActivation
                - (
                    l_n_a *
                    celulas[CELULAS_N_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
                    celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                )
                - (
                    l_ma_a *
                    celulas[CELULAS_MA_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
                    celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                )
                + (
                    d_a *
                    Laplaciano(
                        CELULAS_A_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
            ) * deltaT
            + celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );

    // Macrofago em repouso
    celulas[CELULAS_MR_OFFSET + CELULAS_NOVO_VALOR_OFFSET] =
        max(
            0.0f,
            (
                (-1 * m_mr * celulas[CELULAS_MR_OFFSET + CELULAS_POSICAO_OR_OFFSET])
                - maActivation
                + (
                    (
                        (
                            (MrPMax - MrPMin) *
                            celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] /
                            (
                                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                                + keqch
                            )
                        )
                        + MrPMin
                    )
                    *
                    (
                        MrMaxTissue
                        - (
                            celulas[CELULAS_MR_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                            + celulas[CELULAS_MA_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                        )
                    )
                )
                + (
                    d_mr *
                    Laplaciano(
                        CELULAS_MR_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
                - (
                    x_mr *
                    Quimiotaxia(
                        CELULAS_MR_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
            ) * deltaT
            + celulas[CELULAS_MR_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );

    // Macrofago ativado
    celulas[CELULAS_MA_OFFSET + CELULAS_NOVO_VALOR_OFFSET] =
        max(
            0.0f,
            (
                (-1 * m_ma * celulas[CELULAS_MA_OFFSET + CELULAS_POSICAO_OR_OFFSET])
                + maActivation
                + (
                    d_ma *
                    Laplaciano(
                        CELULAS_MA_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
                - (
                    x_ma *
                    Quimiotaxia(
                        CELULAS_MA_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
            ) * deltaT
            + celulas[CELULAS_MA_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );

    // Citocina Pro-inflamatoria
    celulas[CELULAS_CH_OFFSET + CELULAS_NOVO_VALOR_OFFSET] =
        max(
            0.0f,
            (
                (-1 * m_ch * celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET])
                + (
                    (
                        b_ch_n *
                        celulas[CELULAS_N_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
                        celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                    )
                    +
                    (
                        b_ch_ma *
                        celulas[CELULAS_MA_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
                        celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                    )
                )
                *
                (
                    1
                    - celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] / chInf
                )
                + (
                    d_ch *
                    Laplaciano(
                        CELULAS_CH_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
            ) * deltaT
            + celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );

    // Neutrofilos
    float sourceN =
        (
            (Pmax - Pmin) *
            celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET] /
            (
                celulas[CELULAS_CH_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                + keqch
            )
            + Pmin
        )
        *
        (
            NMaxTissue
            - celulas[CELULAS_N_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );

    celulas[CELULAS_N_OFFSET + CELULAS_NOVO_VALOR_OFFSET] =
        max(
            0.0f,
            (
                (-1 * m_n * celulas[CELULAS_N_OFFSET + CELULAS_POSICAO_OR_OFFSET])
                - (
                    l_n_a *
                    celulas[CELULAS_N_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
                    celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                )
                + sourceN
                + (
                    d_n *
                    Laplaciano(
                        CELULAS_N_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
                - (
                    x_n *
                    Quimiotaxia(
                        CELULAS_N_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
            ) * deltaT
            + celulas[CELULAS_N_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );

    // Neutrofilo apoptotico
    celulas[CELULAS_ND_OFFSET + CELULAS_NOVO_VALOR_OFFSET] =
        max(
            0.0f,
            (
                (
                    m_n *
                    celulas[CELULAS_N_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                )
                +
                (
                    l_n_a *
                    celulas[CELULAS_N_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
                    celulas[CELULAS_A_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                )
                -
                (
                    l_nd_ma *
                    celulas[CELULAS_ND_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
                    celulas[CELULAS_MA_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                )
                + (
                    d_nd *
                    Laplaciano(
                        CELULAS_ND_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
            ) * deltaT
            + celulas[CELULAS_ND_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );

    // Granulo proteico
    celulas[CELULAS_G_OFFSET + CELULAS_NOVO_VALOR_OFFSET] =
        max(
            0.0f,
            (
                (-1 * m_g * celulas[CELULAS_G_OFFSET + CELULAS_POSICAO_OR_OFFSET])
                +
                (
                    b_g_n *
                    sourceN *
                    (
                        1
                        - celulas[CELULAS_G_OFFSET + CELULAS_POSICAO_OR_OFFSET] / gInf
                    )
                )
                + (
                    d_g *
                    Laplaciano(
                        CELULAS_G_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
            ) * deltaT
            + celulas[CELULAS_G_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );

    // Citocina Anti-inflamatoria
    //
    // CORRIGIDO:
    // Antes estava:
    //     d_ca * Laplaciano(CELULAS_A_OFFSET, ...)
    //
    // Pela formulação, difCA = dCA * Delta(CA).
    // Logo, o correto é:
    //     d_ca * Laplaciano(CELULAS_CA_OFFSET, ...)
    celulas[CELULAS_CA_OFFSET + CELULAS_NOVO_VALOR_OFFSET] =
        max(
            0.0f,
            (
                (-1 * m_ca * celulas[CELULAS_CA_OFFSET + CELULAS_POSICAO_OR_OFFSET])
                +
                (
                    (
                        b_ca_mr *
                        celulas[CELULAS_MR_OFFSET + CELULAS_POSICAO_OR_OFFSET] *
                        celulas[CELULAS_ND_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                    )
                    +
                    (
                        b_ca_ma *
                        celulas[CELULAS_MA_OFFSET + CELULAS_POSICAO_OR_OFFSET]
                    )
                )
                *
                (
                    1
                    - celulas[CELULAS_CA_OFFSET + CELULAS_POSICAO_OR_OFFSET] / caInf
                )
                + (
                    d_ca *
                    Laplaciano(
                        CELULAS_CA_OFFSET,
                        celulas,
                        xPosicaoGlobal,
                        yPosicaoGlobal,
                        zPosicaoGlobal,
                        parametrosMalhaGPU
                    )
                )
            ) * deltaT
            + celulas[CELULAS_CA_OFFSET + CELULAS_POSICAO_OR_OFFSET]
        );
}


__kernel void ProcessarPontos(
    __global float *malhaPrincipalAtual,
    __global float *malhaPrincipalAnterior,
    __constant int *parametrosMalhaGPU)
{
    int globalThreadID = get_global_id(0);

    float celulas[CELULAS_SIZEOF];

    // Descobrir posicao 3D local na malha.
    int posicaoGlobalZ =
        globalThreadID /
        (
            parametrosMalhaGPU[MALHA_DIMENSAO_Y] *
            parametrosMalhaGPU[MALHA_DIMENSAO_X]
        );

    int posicaoGlobalY =
        (
            globalThreadID %
            (
                parametrosMalhaGPU[MALHA_DIMENSAO_Y] *
                parametrosMalhaGPU[MALHA_DIMENSAO_X]
            )
        )
        /
        parametrosMalhaGPU[MALHA_DIMENSAO_X];

    int posicaoGlobalX =
        (
            globalThreadID %
            (
                parametrosMalhaGPU[MALHA_DIMENSAO_Y] *
                parametrosMalhaGPU[MALHA_DIMENSAO_X]
            )
        )
        %
        parametrosMalhaGPU[MALHA_DIMENSAO_X];

    if (posicaoGlobalZ >= parametrosMalhaGPU[MALHA_DIMENSAO_Z])
    {
        return;
    }

    //**************************************
    // Preencher celulas para calcular EDOs.
    //**************************************

    int malhaIndex =
        (posicaoGlobalZ * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z])
        +
        (posicaoGlobalY * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y])
        +
        (posicaoGlobalX * parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X]);

    // Loop por todas as celulas.
    for (int count = 0; count < MALHA_TOTAL_CELULAS; count++)
    {
        int celulaLocalOffset =
            (CELULA_A + count) * CELULAS_SINGLE_CELL_SIZEOF;

        int celulaMalhaOffset =
            (CELULA_A + count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS];

        // Origem.
        celulas[celulaLocalOffset + CELULAS_POSICAO_OR_OFFSET] =
            malhaPrincipalAnterior[
                malhaIndex
                + celulaMalhaOffset
            ];

        // Vizinhanca Z+.
        celulas[celulaLocalOffset + CELULAS_POSICAO_ZP_OFFSET] =
            (posicaoGlobalZ + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_Z])
            ?
            malhaPrincipalAnterior[
                malhaIndex
                + celulaMalhaOffset
                + parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z]
            ]
            :
            0.0f;

        // Vizinhanca Z-.
        celulas[celulaLocalOffset + CELULAS_POSICAO_ZM_OFFSET] =
            (posicaoGlobalZ - 1 >= 0)
            ?
            malhaPrincipalAnterior[
                malhaIndex
                + celulaMalhaOffset
                - parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Z]
            ]
            :
            0.0f;

        // Vizinhanca Y+.
        celulas[celulaLocalOffset + CELULAS_POSICAO_YP_OFFSET] =
            (posicaoGlobalY + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_Y])
            ?
            malhaPrincipalAnterior[
                malhaIndex
                + celulaMalhaOffset
                + parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y]
            ]
            :
            0.0f;

        // Vizinhanca Y-.
        celulas[celulaLocalOffset + CELULAS_POSICAO_YM_OFFSET] =
            (posicaoGlobalY - 1 >= 0)
            ?
            malhaPrincipalAnterior[
                malhaIndex
                + celulaMalhaOffset
                - parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_Y]
            ]
            :
            0.0f;

        // Vizinhanca X+.
        celulas[celulaLocalOffset + CELULAS_POSICAO_XP_OFFSET] =
            (posicaoGlobalX + 1 < parametrosMalhaGPU[MALHA_DIMENSAO_X])
            ?
            malhaPrincipalAnterior[
                malhaIndex
                + celulaMalhaOffset
                + parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X]
            ]
            :
            0.0f;

        // Vizinhanca X-.
        celulas[celulaLocalOffset + CELULAS_POSICAO_XM_OFFSET] =
            (posicaoGlobalX - 1 >= 0)
            ?
            malhaPrincipalAnterior[
                malhaIndex
                + celulaMalhaOffset
                - parametrosMalhaGPU[MALHA_DIMENSAO_POSICAO_X]
            ]
            :
            0.0f;
    }

    //**************************************
    // Atualizar malha com pontos calculados.
    //**************************************

    CalcularPontos(
        celulas,
        posicaoGlobalX,
        posicaoGlobalY,
        posicaoGlobalZ,
        parametrosMalhaGPU
    );

    // Loop por todas as celulas.
    for (int count = 0; count < MALHA_TOTAL_CELULAS; count++)
    {
        int celulaLocalOffset =
            (CELULA_A + count) * CELULAS_SINGLE_CELL_SIZEOF;

        int celulaMalhaOffset =
            (CELULA_A + count) * parametrosMalhaGPU[MALHA_DIMENSAO_CELULAS];

        malhaPrincipalAtual[
            malhaIndex
            + celulaMalhaOffset
        ] =
            celulas[
                celulaLocalOffset
                + CELULAS_NOVO_VALOR_OFFSET
            ];
    }
}