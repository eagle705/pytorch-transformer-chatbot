from tqdm import tqdm
import torch
from metric import correct_sum
from chatspace import ChatSpace

spacer = ChatSpace()

def evaluate(model, data_loader, metrics, device, tokenizer=None):
    if model.training:
        model.eval()

    summary = {metric: 0 for metric in metrics}
    num_correct_elms = 0

    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        enc_input, dec_input, dec_output = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_pred = model(enc_input, dec_input)

            if step % 1000 == 0:
                decoding_from_result(enc_input, y_pred, dec_output, tokenizer)

            y_pred = y_pred.reshape(-1, y_pred.size(-1))
            dec_output = dec_output.view(-1).long()

            for metric in metrics:
                if metric is 'acc':
                    _correct_sum, _num_correct_elms = correct_sum(y_pred, dec_output)
                    summary[metric] += _correct_sum
                    num_correct_elms += _num_correct_elms
                else:
                    summary[metric] += metrics[metric](y_pred, dec_output).item() #* dec_output.size()[0]

    for metric in metrics:
        if metric is 'acc':
            summary[metric] /= num_correct_elms
        else:
            summary[metric] /= len(data_loader.dataset)

    return summary

def decoding_from_result(enc_input, y_pred, dec_output=None, tokenizer=None):
    list_of_input_ids = enc_input.tolist()
    list_of_pred_ids = y_pred.max(dim=-1)[1].tolist()
    input_token = tokenizer.decode_token_ids(list_of_input_ids)
    pred_token = tokenizer.decode_token_ids(list_of_pred_ids)

    print("input: ", input_token)
    print("pred: ", pred_token)
    if dec_output is not None:
        real_token =  tokenizer.decode_token_ids(dec_output.tolist())
        print("real: ", real_token)
        print("")
        return None
    else:
        # 핑퐁의 띄어쓰기 교정기 적용
        pred_str = ''.join([token.split('/')[0] for token in pred_token[0][:-1]])
        pred_str = spacer.space(pred_str)
        print("pred_str: ", pred_str)
        print("")
        return pred_str

