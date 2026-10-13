# Auditoria de Calçadas - YOLOv8n com inferência mobile

Aplicação web experimental para demonstrar detecção de seis classes urbanas com o modelo YOLOv8n do checkpoint integrado CP2 + CP3 da FIAP. A inferência roda no navegador com ONNX Runtime Web/WASM.

## Objetivo e limites

O projeto explora a identificação de alguns elementos em cenários urbanos pela câmera traseira de um smartphone. É experimental, pode cometer erros e não substitui vistoria humana. As classes não cobrem uma auditoria completa de acessibilidade.

## Tecnologias e arquitetura

- Vanilla JavaScript + Vite, mantendo a base de [Mjrovai/yoloinferApp](https://github.com/Mjrovai/yoloinferApp).
- ONNX Runtime Web 1.18.0, backend WASM com SIMD e uma thread.
- Câmera via getUserMedia, preferência facingMode environment; Safari requer contexto seguro HTTPS.
- Pré-processamento compatível com o fluxo padrão Ultralytics: pixels RGB (canvas não usa BGR), letterbox central 640 × 640 com padding cinza 114 e arredondamento das bordas como no exemplo ONNX oficial; entrada NCHW float32 normalizada em 0–1.
- Contrato observado no artefato presente: entrada [1,3,640,640], saída [1,10,8400]. Canais 0–3: centro x/y e largura/altura; canais 4–9: scores das seis classes. Aplica limiar configurável e NMS por classe; converte coordenadas do letterbox e desenha caixas, classe e confiança.
- O carregamento executa inferência de aquecimento e rejeita saída incompatível.

## Classes

| ID | Nome | Rótulo |
|---:|---|---|
| 0 | animal | Animal |
| 1 | traffic_lights | Semáforo |
| 2 | waste_container | Contentor de resíduos |
| 3 | pothole | Buraco |
| 4 | cracks | Rachadura |
| 5 | open_manhole | Bueiro aberto |

## Dataset e treinamento informado

- 2.518 imagens originalmente em sete categorias; good_road foi excluída porque continha 100 anotações vazias.
- Divisão: 1.933 treino, 241 validação, 244 teste.
- Ultralytics YOLOv8n, GPU Tesla T4, 15 épocas, resolução de treino 640 × 640.
- Foi informada a existência de uma exportação 192 × 192 com saída [1,10,756], mas o arquivo best.onnx recebido foi validado como modelo 640 × 640. O ONNX Runtime Web mediu entrada [1,3,640,640] e saída [1,10,8400]; 192 foi rejeitada pelo runtime, que reportou dimensão fixa 640. O ONNX está em public/models/auditoria_640.onnx (12.778.116 bytes); a cópia idêntica mantida na raiz foi removida. O app valida a forma de saída ao carregar.

Métricas medidas no teste informado:

| Métrica | Resultado |
|---|---:|
| Precision | 0,7412 |
| Recall | 0,6572 |
| mAP@50 | 0,6958 |
| mAP@50–95 | 0,3970 |

| Classe | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| animal | 0,698 | 0,512 | 0,577 |
| traffic_lights | 0,711 | 0,409 | 0,473 |
| waste_container | 0,800 | 0,843 | 0,865 |
| pothole | 0,774 | 0,762 | 0,794 |
| cracks | 0,740 | 0,542 | 0,558 |
| open_manhole | 0,726 | 0,875 | 0,908 |

Esses valores são do conjunto de teste de treinamento, não garantias no celular. Luz, ângulo, escala, câmera e limiar afetam os resultados.

## Estrutura

- index.html
- src/main.js: câmera, estado e desenho
- src/inference.js: modelo, pré/pós-processamento e NMS
- src/style.css
- public/models/auditoria_640.onnx
- public/manifest.json, public/sw.js e ícones PNG referenciados pela PWA
- classes.yaml, vite.config.js, netlify.toml
- LICENSE-original.txt: licença MIT da base


## Instalação e execução

Requer Node.js 18+ e npm.

    npm ci
    npm run dev

Vite com basicSsl fornece URL HTTPS local e de rede; computador e celular precisam compartilhar a rede. Certificado autoassinado pode exigir aceitação. Build e preview:

    npm run build
    npm run preview -- --host 0.0.0.0


## Safari no iPhone

1. Abra no Safari o domínio HTTPS produzido pelo deploy. Para desenvolvimento, use URL HTTPS do Vite com computador e celular na mesma rede; certificado autoassinado é apenas para teste.
2. Aguarde “Modelo carregado · 6 classes” (download inicial ~12,8 MB).
3. Toque Iniciar detecção e autorize câmera. A aplicação solicita câmera traseira como preferência.
4. Confirme que o status mostra “Modelo carregado · 6 classes”. Inicie, aponte para objetos das classes ensinadas e confira caixas, nomes e confiança; toque Parar detecção para liberar câmera.
5. Opcional: use Adicionar à Tela de Início.


## Substituir o modelo

O caminho padrão é public/models/auditoria_640.onnx, com entrada [1,3,640,640] e saída [1,10,8400] verificadas para o artefato incluído. A exportação 192 × 192 descrita no enunciado não estava disponível; o best.onnx recebido era o modelo 640 e sua cópia duplicada foi removida após validação. Preserve o modelo antes de substituí-lo. Se trocar modelo/resolução, ajuste src/inference.js e valide pré/pós-processamento usando o runtime. Limpe cache após atualização. Não há URL de download porque nenhuma origem verificável foi fornecida.

## Problemas comuns

- Modelo ausente/incompatível: confira caminho e formas esperadas, console e painel de rede.
- Câmera não abre: HTTPS (ou localhost), permissão concedida e câmera sem uso por outro app.
- Permissão negada: habilite câmera nas configurações de site do iOS e recarregue.
- Tela preta/cache antigo: confirme contexto seguro e página em primeiro plano; recarregue sem cache ou limpe dados do site.
- Poucas detecções: reduza confiança, melhore luz/distância e confira classes/export.

## Créditos, licença e equipe

Base: [Mjrovai/yoloinferApp](https://github.com/Mjrovai/yoloinferApp), Marcelo Rovai. A licença original MIT foi preservada em LICENSE-original.txt. O modelo é artefato separado; confirme licença do modelo/dataset antes de redistribuir.

- Guilherme Doretto Sobreiro — RM: 996764
- Guilherme Fazito Zioli — RM: 550539

Confirme termos do dataset/modelo e preencha equipe antes da entrega. A origem específica do dataset não foi identificada nos arquivos locais.

## Validação neste checkout

- Inspecionados código original, dependências, fluxo de câmera, pré/pós-processamento e licença.
- npm ci executado. npm run build aprovado (Vite 6.4.4); permanece o aviso de tamanho do bundle e eval na dependência ONNX Runtime.
- npm test aprovado: 6 testes para IDs das classes, decodificação channel-major, limiar de confiança, NMS por classe, conversão de coordenadas letterbox, arredondamento Ultralytics e rejeição de forma incorreta.
- Vulnerabilidades encontradas no audit inicial e resolvidas:


  A atualização do plugin TLS para @vitejs/plugin-basic-ssl 1.2.0 foi compatibilidade necessária para Vite 6. Rodei npm audit fix sem --force para as dependências transitivas; o audit final encontrou zero vulnerabilidades.
- ONNX Runtime Web 1.18.0 executou inferência de aquecimento: [1,3,640,640] → [1,10,8400]. Uma tentativa com 192 × 192 foi rejeitada, confirmando que o arquivo disponível é de resolução 640. A saída [1,10,8400] foi validada. Nenhuma imagem de teste do dataset nem pesos PyTorch foi encontrada: não há comparação de previsões com imagens reais nem com PyTorch.
- No tensor de aquecimento totalmente preto, o maior score de classe ficou abaixo de 0,001. Isso confirma execução técnica do grafo, não qualidade de detecção em cenas reais.
- Os créditos ao projeto base e a licença MIT estão preservados neste README e em LICENSE-original.txt.

