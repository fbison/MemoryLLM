import sys
import torch
from transformers import AutoTokenizer
from modeling_mplus import MPlus

def carregar_modelo():
    print("Carregando o modelo MPlus-8B... Por favor, aguarde.")
    model = MPlus.from_pretrained(
        "YuWangX/mplus-8b", 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("YuWangX/mplus-8b")
    model = model.to(torch.bfloat16)
    model.put_ltm_to_numpy()
    print("Modelo carregado com sucesso!\n")
    return model, tokenizer

def read_memory():
    print("\nCole o texto abaixo.")
    print("Quando terminar, digite uma linha contendo apenas:")
    print("<<<END>>>\n")

    linhas = []

    while True:
        try:
            linha = sys.stdin.readline()

            if not linha:
                break

            if linha.strip() == "<<<END>>>":
                break

            linhas.append(linha)

        except KeyboardInterrupt:
            break

    return "".join(linhas)

def main():
    model, tokenizer = carregar_modelo()
    memorias_armazenadas = []

    while True:
        print("\n=== Menu MPlus ===")
        print("1 - Adicionar memória")
        print("2 - Fazer pergunta")
        print("3 - Mostrar memórias")
        print("4 - Deletar memória")
        print("5 - Sair")
        
        escolha = input("\nEscolha uma opção: ")

        if escolha == '1':
            
            texto_memoria = read_memory()
            if len(texto_memoria.split()) < 10:
                print("[Aviso] Textos muito curtos (menos de ~16 tokens) podem instabilizar a memória do modelo.")
            
            input_ids = tokenizer(texto_memoria, return_tensors='pt', add_special_tokens=False).input_ids.cuda()

            model.inject_memory(input_ids, update_memory=True)
            memorias_armazenadas.append(texto_memoria)
            print("Memória adicionada e injetada com sucesso!")

        elif escolha == '2':
            pergunta = input("\nDigite sua pergunta: ")
            # Formatação simples estilo QA
            prompt = f"Question: {pergunta} Answer:"
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
            
            print("Gerando resposta...")
            outputs = model.generate(input_ids=inputs, max_new_tokens=100)
            resposta = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            print(f"\n[Resposta do Modelo]: {resposta.strip()}")

        elif escolha == '3':
            print("\n--- Memórias Atuais ---")
            if not memorias_armazenadas:
                print("Nenhuma memória foi adicionada ainda.")
            else:
                for i, mem in enumerate(memorias_armazenadas):
                    print(f"[{i}] {mem}")
            print("-----------------------")

        elif escolha == '4':
            print("\n--- Deletar Memória ---")
            if not memorias_armazenadas:
                print("Nenhuma memória para deletar.")
                continue
                
            for i, mem in enumerate(memorias_armazenadas):
                print(f"[{i}] {mem[:50]}...")
                
            try:
                idx = int(input("Digite o índice da memória que deseja deletar: "))
                if 0 <= idx < len(memorias_armazenadas):
                    removida = memorias_armazenadas.pop(idx)
                    print(f"Memória removida do registro local: '{removida[:30]}...'")
                    print("[Nota] A arquitetura do MPlus funde as memórias nos pesos dinâmicos. "
                          "Para limpar completamente a memória da rede, seria necessário reiniciar o script.")
                else:
                    print("Índice inválido.")
            except ValueError:
                print("Entrada inválida. Digite um número.")

        elif escolha == '5':
            print("Encerrando o sistema MPlus. Até logo!")
            sys.exit(0)

        else:
            print("Opção inválida, por favor tente novamente.")

if __name__ == "__main__":
    main()
