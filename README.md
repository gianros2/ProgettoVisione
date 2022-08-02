# ProgettoVisione

## Problema
Identificazione e sostituzione di brand utilizzati sui pannelli pubblicitari nel contesto delle partite di calcio.<br>
Il progetto prevede l'applicazione di algoritmi e concetti legati alla Computer Vision, i quali sono stati sviluppati, nello specifico,<br> 
per la detection dei loghi Adidas, Fedex e Ps3 con successiva sostituzione di questi ultimi con il logo DHL.        

[![problem.gif](https://i.postimg.cc/8PdRtfB8/problem.gif)](https://postimg.cc/D8Z4zZM5)

## Dataset

<table>
    <tr>
        <th> Class </th>
        <th> n. Images </td>
        <th> Istanze </td>
    </tr>
    <tr>
        <td rowspan="2">Adidas</th>
        <td>Train: 400</td>
        <td>3181</td>
    </tr>
    <tr>
        <td>Cv: 130</td>
        <td>1090</td>
    </tr>
    <tr>
        <td rowspan="2">Fedex</th>
        <td>Train: 400</td>
        <td>1960</td>
    </tr>
    <tr>
        <td>Cv: 130</td>
        <td>578</td>
    </tr>
    <tr>
        <td rowspan="2">PS3</th>
        <td>Train: 400</td>
        <td>692</td>
    </tr>
    <tr>
        <td>Cv: 130</td>
        <td>229</td>
    </tr>
</table>

## Risultati

[![result.jpg](https://i.postimg.cc/hv10zX2q/result.jpg)](https://postimg.cc/y3WR28qf)

https://user-images.githubusercontent.com/105881522/182245394-858e02d8-1f2a-4048-b62e-562fd30b7745.mp4

## Getting Started

Clonare la repository del progetto mediante il comando:
- **git clone** https://github.com/gianros2/ProgettoVisione

Estrarre i pesi nella directory darknet/pesi/people-detection partendo dal file **yolo-people-detection.part1.rar**

Posizionarsi nella directory ProgettoVisione/detection e sostituzione ed eseguire il comando seguente:
- **pip install -r requirements.txt** (per installare tutte le lib necessarie al funzionamento del progetto)

[IMAGE]<br>
Impostare la variabile **image_path** al percorso contentenente l'immagine da elaborare e la variabile **on_video** su **False**.

[VIDEO]<br>
Impostare la variabile **video_path** al percorso contentenente il video da elaborare e la variabile **on_video** su **True**.

Impostare la variabile **confidence_threshold** sulla soglia di detection dei loghi voluta.<br>
Impostare la variabile **people_confidence** sulla soglia di detection dei giocatori voluta.

Avviare lo script **script_detection_e_sostituzione.py** tramite un IDE o da linea di comando eseguendo:
- **python script_detection_e_sostituzione.py**
