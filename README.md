# F5-TTS_scripts
Prepare F5-TTS anaconda enviroment:
```
conda create -n f5-tts python=3.10 && conda activate f5-tts
conda install git
```
# NVIDIA GPU: install pytorch with your CUDA version, e.g.
```
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
# AMD GPU: install pytorch with your ROCm version, e.g.
```
pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
```
```
pip install git+https://github.com/SWivid/F5-TTS.git

git clone https://github.com/sarpba/F5-TTS_scripts.git
cd F5-TTS_scripts
pip install -r requirements.txt
```

script usage:
```
usage: f5_tts_infer_API.py [-h] -i INPUT_DIR -ig INPUT_GEN_DIR -o OUTPUT_DIR
                           [--remove_silence] --vocab_file VOCAB_FILE
                           --ckpt_file CKPT_FILE [--speed SPEED]
                           [--nfe_step NFE_STEP] [--max_workers MAX_WORKERS]
                           [--norm NORM] [--seed SEED]

F5-TTS Batch Inference Script with Multi-GPU Support

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Input directory containing .wav and corresponding .txt
                        files (reference texts)
  -ig INPUT_GEN_DIR, --input_gen_dir INPUT_GEN_DIR
                        Input generation directory containing .txt files
                        (generated texts) with the same names as input .wav
                        files
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory to save generated .wav files
  --remove_silence      Remove silence from generated audio
  --vocab_file VOCAB_FILE
                        Path to the vocabulary file (e.g., /path/to/vocab.txt)
  --ckpt_file CKPT_FILE
                        Path to the model checkpoint file (e.g.,
                        /path/to/model.pt)
  --speed SPEED         Speed of the generated audio (0.3-2.0). Default is 1.0
  --nfe_step NFE_STEP   Number of NFE steps (16-64). Default is 32
  --max_workers MAX_WORKERS
                        Maximum number of parallel workers. Defaults to the
                        number of available GPUs.
  --norm NORM           Normalization type (e.g., 'hun', 'eng'). Determines
                        which normalizer to use.
  --seed SEED           Random seed for reproducibility. Default is -1, which
                        selects a random seed.
```
example:
```
cd scripts
python f5_tts_infer_API.py -i INPUT_DIR -ig INPUT_GEN_DIR -o OUTPUT_DIR --vocab_file hun_v4_vocab.txt --ckpt_file model_349720.pt --norm hun
```


## Bevezetés

Az **F5-TTS** egy többszálas (multi-GPU) beszédszintézis megoldás, amely egy előtanított neurális hálózatot használ szövegből hang generálására. Ez a script lehetővé teszi a referenciahangok és referencia-szövegek alapján történő szintézist.

## Telepítési követelmények


A csomagok telepíthetők az alábbi parancsok futtatásával:

F5-TTS anaconda környezet kialakítása:
```
conda create -n f5-tts python=3.10 && conda activate f5-tts
conda install git
```
# NVIDIA GPU: install pytorch with your CUDA version, e.g.
```
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
# AMD GPU: install pytorch with your ROCm version, e.g.
```
pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
```
```
pip install git+https://github.com/SWivid/F5-TTS.git

git clone https://github.com/sarpba/F5-TTS_scripts.git
cd F5-TTS_scripts
pip install -r requirements.txt
```

## Használat
```
cd scripts
python f5_tts_infer_API.py -i INPUT_DIR -ig INPUT_GEN_DIR -o OUTPUT_DIR --vocab_file hun_v4_vocab.txt --ckpt_file model_349720.pt --norm hun
```

### Argumentumok magyarázata

| Argumentum | Leírás |
|------------|--------|
| `--input_dir` | Az a könyvtár, amely tartalmazza a referenciahangokat (`.wav` fájlokat) és a referencia-szövegeket (`.txt` fájlokat). |
| `--input_gen_dir` | Az a könyvtár, amely tartalmazza a generált szövegeket (`.txt` fájlok). A fájlneveknek egyezniük kell a referenciahangokéval. |
| `--output_dir` | Az a könyvtár, ahová a generált beszédfájlok mentésre kerülnek. |
| `--vocab_file` | A szótár (`.txt` formátumban), amelyet a modell használ. |
| `--ckpt_file` | Az előtanított modell ellenőrzőpontja (`.pt` fájl). |
| `--speed` | A beszéd sebessége (0.3 és 2.0 között). Alapértelmezett: `1.0`. |
| `--nfe_step` | Az inference lépéseinek száma (16 és 64 között). Alapértelmezett: `32`. |
| `--max_workers` | A párhuzamosan futó folyamatok száma. Alapértelmezés szerint a rendelkezésre álló GPU-k száma. |
| `--norm` | Normalizáció típusa (`hun` a magyar nyelvhez, `eng` az angol nyelvhez). |
| `--seed` | A véletlenszám-generátor magja a reprodukálhatóság érdekében. Alapértelmezés szerint véletlenszerű. |

## A script működése

### 1. A fájlok betöltése és ellenőrzése
- A script beolvassa a `--input_dir` és `--input_gen_dir` könyvtárban található fájlokat.
- Ellenőrzi, hogy minden referenciahanghoz és generált szöveghez tartozik-e megfelelő `.txt` fájl.

### 2. A modellek betöltése
- Betölti a `DiT` vagy `UNetT` modellt az `--ckpt_file` ellenőrzőpont segítségével.
- Betölti a `vocos` vocoder modellt.

### 3. Inferencia végrehajtása
- A referenciahang és a referencia-szöveg alapján a generált szöveget szintetizálja.
- Ha szükséges, a kimeneti fájlból eltávolítja a csendet (`--remove_silence`).

### 4. Többszálú végrehajtás
- A script automatikusan felismeri a rendelkezésre álló GPU-k számát és azok között párhuzamosan osztja el a feladatokat.
- Több GPU esetén a `multiprocessing` modult használja az inferencia gyorsítására.

## Magyar nyelvű normalizáló működése

A script tartalmaz egy **magyar nyelvi normalizálót**, amely különböző átalakításokat végez a szövegen:

- **Erőltetett cserék**: Egy `force_changes.csv` fájl alapján előre meghatározott szöveghelyettesítéseket végez.
- **Alap cserék**: Egy `changes.csv` fájlból származó szavakat és kifejezéseket módosít.
- **Sorszámok kezelése**: Az arab számokból szöveges formátumot állít elő.
- **Dátumok átalakítása**: Az év, hónap, nap formátumokat felismeri és átalakítja szöveges formára.
- **Időpontok kezelése**: Az `HH:MM` és `HH:MM:SS` formátumokat felismeri és szövegesen átírja.
- **Számok szöveggé alakítása**: Az arab számokat szavakká konvertálja.
- **Felesleges karakterek eltávolítása**: Eltávolítja a nem kívánt speciális karaktereket.
- **Többszörös szóközök eltávolítása**: Megtisztítja a szöveget a felesleges szóközöktől.
- **Előtag hozzáadása**: A szöveg elejéhez hozzáfűzi a "..." jelet.

A normalizáló célja a magyar nyelvű beszédszintézis javítása azáltal, hogy a bemeneti szöveget természetesebb formátumra alakítja át.

## Összegzés
Ez a script egy erőteljes eszköz az F5-TTS modell többszálú inferenciájához. Segítségével gyorsan és hatékonyan lehet nagy mennyiségű szöveget beszéddé alakítani. A párhuzamos végrehajtásnak köszönhetően kihasználja a több GPU adta lehetőségeket, és lehetőséget biztosít a beszédsebesség, normalizáció és egyéb paraméterek finomhangolására.

