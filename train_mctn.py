import torch
from fusion import Seq2Seq, L2MCTN
from eval_metrics import eval_mosei_senti_return
from torch.nn import functional as F
from torch import nn


def train(
        traindata, validdata,
        encoder0, decoder0, encoder1, decoder1,
        reg_encoder, head,
        criterion_t0=nn.MSELoss(), criterion_c=nn.MSELoss(),
        criterion_t1=nn.MSELoss(), criterion_r=nn.L1Loss(),
        max_seq_len=20,
        mu_t0=0.01, mu_c=0.01, mu_t1=0.01,
        dropout_p=0.1, early_stop=False, patience_num=15,
        lr=1e-4, weight_decay=0.01, op_type=torch.optim.AdamW,
        epoch=100, model_save='best_mctn.pt'):
    seq2seq0 = Seq2Seq(encoder0, decoder0).cuda()
    seq2seq1 = Seq2Seq(encoder1, decoder1).cuda()
    model = L2MCTN(seq2seq0, seq2seq1, reg_encoder, head, p=dropout_p).cuda()
    op = op_type(model.parameters(), lr=lr, weight_decay=weight_decay)

    patience = 0
    best_mae = 10000

    for ep in range(epoch):
        model.train()
        print('start training ---------->>')

        sum_total_loss = 0
        sum_reg_loss = 0
        total_batch = 0
        for inputs in traindata:
            src, trg0, trg1, labels, _ = _process_input_L2(
                inputs, max_seq_len)
            translation_loss_0 = 0
            cyclic_loss = 0
            translation_loss_1 = 0
            op.zero_grad()
            out, reout, rereout, head_out = model(src, trg0, trg1)

            for j, o in enumerate(out):
                translation_loss_0 += criterion_t0(o, trg0[j])
            translation_loss_0 = translation_loss_0 / out.size(0)

            for j, o in enumerate(reout):
                cyclic_loss += criterion_c(o, src[j])
            cyclic_loss = cyclic_loss / reout.size(0)

            for j, o in enumerate(rereout):
                translation_loss_1 += criterion_t1(o, trg1[j])
            translation_loss_1 = translation_loss_1 / rereout.size(0)

            reg_loss = criterion_r(head_out, labels)

            total_loss = mu_t0 * translation_loss_0 + mu_c * \
                cyclic_loss + mu_t1 * translation_loss_1 + reg_loss

            sum_total_loss += total_loss
            sum_reg_loss += reg_loss
            total_batch += 1

            total_loss.backward()
            op.step()

        sum_total_loss /= total_batch
        sum_reg_loss /= total_batch

        print('Train Epoch {}, total loss: {}, regression loss: {}, embedding loss: {}'
              .format(ep, sum_total_loss, sum_reg_loss, sum_total_loss - sum_reg_loss))

        model.eval()
        print('Start Evaluating ---------->>')
        pred = []
        true = []
        with torch.no_grad():
            for inputs in validdata:
                # process input
                src, trg0, trg1, labels, _ = _process_input_L2(
                    inputs, max_seq_len)

                #  We only need the source text as input! No need for target!
                _, _, _, head_out = model(src)
                pred.append(head_out)
                true.append(labels)

            eval_results_include = eval_mosei_senti_return(
                torch.cat(pred, 0), torch.cat(true, 0), exclude_zero=False)
            eval_results_exclude = eval_mosei_senti_return(
                torch.cat(pred, 0), torch.cat(true, 0), exclude_zero=True)
            mae = eval_results_include[0]
            acc1 = eval_results_include[-1]
            acc2 = eval_results_exclude[-1]
            print('Eval Epoch: {}, MAE: {}, Acc1: {}, Acc2: {}'.format(
                ep, mae, acc1, acc2))

            if mae < best_mae:
                patience = 0
                best_mae = mae
                print('<------------ Saving Best Model')
                print()
                torch.save(model, model_save)
            else:
                patience += 1
            if early_stop and patience > patience_num:
                break


def single_test(model, testdata, max_seq_len=20):
    model.eval()
    print('Start Testing ---------->>')
    pred = []
    true = []
    with torch.no_grad():
        for inputs in testdata:
            # process input
            src, _, _, labels, _ = _process_input_L2(inputs, max_seq_len)

            #  We only need the source text as input! No need for target!
            _, _, _, head_out = model(src)
            pred.append(head_out)
            true.append(labels)

        eval_results_include = eval_mosei_senti_return(
            torch.cat(pred, 0), torch.cat(true, 0), exclude_zero=False)
        eval_results_exclude = eval_mosei_senti_return(
            torch.cat(pred, 0), torch.cat(true, 0), exclude_zero=True)
        mae = eval_results_include[0]
        acc1 = eval_results_include[-1]
        acc2 = eval_results_exclude[-1]
        print('Test: MAE: {}, Acc1: {}, Acc2: {}'.format(mae, acc1, acc2))
        return {'Acc:': acc2}

def test(model, test_dataloader):
    single_test(model, test_dataloader)


def _process_input_L2(inputs, max_seq=20):
    src = inputs[0][2][:, :max_seq, :]
    trg0 = inputs[0][0][:, :max_seq, :]
    trg1 = inputs[0][1][:, :max_seq, :]
    feature_dim = max(src.size(-1), trg0.size(-1), trg1.size(-1))

    src = F.pad(src, (0, feature_dim - src.size(-1)))
    trg0 = F.pad(trg0, (0, feature_dim - trg0.size(-1)))
    trg1 = F.pad(trg1, (0, feature_dim - trg1.size(-1)))

    src = src.transpose(1, 0).cuda()
    trg0 = trg0.transpose(1, 0).cuda()
    trg1 = trg1.transpose(1, 0).cuda()
    labels = inputs[-1].cuda()

    return src, trg0, trg1, labels, feature_dim
